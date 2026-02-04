#!/usr/bin/env python
# coding: utf-8

# In[66]:


import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

pid = "patient0028"
#pid =  "whoever"
for view in ["2CH", "4CH"]:
    for phase in ["ED", "ES"]:
        img = nib.load(
            f"database_nifti/{pid}/{pid}_{view}_{phase}.nii.gz"
        ).get_fdata()

        lv_mask = get_lv_mask(pid, view=view, phase=phase, source="gt")

        plt.figure(figsize=(3, 3))
        plt.imshow(img, cmap="gray")
        plt.imshow(lv_mask, alpha=0.4)
        plt.title(f"{pid} | {view} | {phase}")
        plt.axis("off")
        plt.show()
def get_lv_mask(pid, view="2CH", phase="ED", source="gt"):
    """
    Returns a boolean LV cavity mask.
    """
    if source == "gt":
        mask = nib.load(
            f"database_nifti/{pid}/{pid}_{view}_{phase}_gt.nii.gz"
        ).get_fdata()
        return mask == 1

# Load image
img = nib.load(
    f"database_nifti/{pid}/{pid}_{view}_{phase}.nii.gz").get_fdata()

# Get LV mask (THIS is the abstraction you wanted)
lv_mask = get_lv_mask(pid, view=view, phase=phase, source="gt")

# Compute area in pixels
lv_area_pixels = lv_mask.sum()
print("LV area (pixels):", lv_area_pixels)

# Visualize overlay
plt.figure(figsize=(4, 4))
plt.imshow(img, cmap="gray")
plt.imshow(lv_mask, alpha=0.4)
plt.title(f"{pid} — {view} {phase}")
plt.axis("off")
plt.show()


# In[67]:


def lv_area_mm2(pid, view="2CH", phase="ED"):
    base = f"database_nifti/{pid}"

    img_nii = nib.load(f"{base}/{pid}_{view}_{phase}.nii.gz")
    mask = nib.load(f"{base}/{pid}_{view}_{phase}_gt.nii.gz").get_fdata()

    # Pixel spacing (mm)
    dx, dy = img_nii.header.get_zooms()[:2]
    pixel_area = dx * dy  # mm²

    lv_pixels = (mask == 1).sum()
    return lv_pixels * pixel_area


# In[68]:


for phase in ["ED", "ES"]:
    area = lv_area_mm2(pid, "2CH", phase)
    print(f"2CH {phase} area (mm²): {area:.1f}")


# In[69]:


def lv_volume_proxy(pid, phase="ED"):
    A2 = lv_area_mm2(pid, "2CH", phase)
    A4 = lv_area_mm2(pid, "4CH", phase)
    return np.sqrt(A2 * A4)
EDV = lv_volume_proxy(pid, "ED")
ESV = lv_volume_proxy(pid, "ES")

print("EDV proxy:", EDV)
print("ESV proxy:", ESV)


# In[70]:


SV = EDV - ESV
print("Stroke volume (proxy units):", SV)
EF = SV / EDV
print("Ejection fraction (proxy):", EF)


# In[71]:


def lv_length_mm(pid, view="4CH", phase="ED"):
    base = f"database_nifti/{pid}"
    img_nii = nib.load(f"{base}/{pid}_{view}_{phase}.nii.gz")
    mask = nib.load(f"{base}/{pid}_{view}_{phase}_gt.nii.gz").get_fdata()
    dx, dy = img_nii.header.get_zooms()[:2]
    ys, xs = np.where(mask == 1)
    # approximate long axis as max distance between LV pixels
    length_pixels = np.sqrt(
        (xs.max() - xs.min())**2 +
        (ys.max() - ys.min())**2 )
    return length_pixels * np.mean([dx, dy])  # mm


# In[72]:


def lv_volume_ml(pid, phase="ED"):
    A2 = lv_area_mm2(pid, "2CH", phase)
    A4 = lv_area_mm2(pid, "4CH", phase)
    L = lv_length_mm(pid, "4CH", phase)

    V_mm3 = (8 / (3 * np.pi)) * (A2 * A4) / L
    return V_mm3 / 1000  # mm³ → mL

EDV = lv_volume_ml(pid, "ED")
ESV = lv_volume_ml(pid, "ES")
SV  = EDV - ESV

print(f"EDV: {EDV:.1f} mL")
print(f"ESV: {ESV:.1f} mL")
print(f"Stroke Volume: {SV:.1f} mL")


# In[73]:


def cardiac_output_L_min(SV_ml, HR_bpm):
    return (SV_ml * HR_bpm) / 1000  # mL/min → L/min
HR = 80  # bpm
CO = cardiac_output_L_min(SV, HR)
print(f"Cardiac Output @ {HR} bpm: {CO:.2f} L/min")


# In[74]:


def lvot_area_cm2(d_cm):
    return np.pi * (d_cm / 2)**2
def vti_equivalent_cm(SV_ml, lvot_d_cm):
    A = lvot_area_cm2(lvot_d_cm)  # cm²
    SV_cm3 = SV_ml  # 1 mL = 1 cm³
    return SV_cm3 / A

d_vals = np.linspace(1.4, 2.2, 9)  # cm

for d in d_vals:
    vti = vti_equivalent_cm(SV, d)
    print(f"d={d:.2f} cm → VTI={vti:.1f} cm")

