This README.txt file was generated on 2025-10-29 by Hanae Elmekki

--------------------
GENERAL INFORMATION
--------------------

1. Title of Dataset:  CACTUS: An open dataset and framework for automated Cardiac Assessment and Classification of Ultrasound images using deep transfer learning. 

2. Author Information
        A. Principal Investigator Contact Information
                Name: Hanae Elmekki
                Institution:  Concordia University
                Email: hanae.elmekki@mail.concordia.ca


        B. Associate or Co-investigator Contact Information
                Name: Amanda Spilkin
                Institution:  Concordia University
                Email: amanda.spilkin@mail.concordia.ca

3. Date of data collection (single date, range, approximate date): 2025-03-05

4. Geographic location of data collection: Concordia University, Montreal, Quebec, Canada

5. Information about funding sources that supported the collection of the data: Natural Sciences and Engineering Research Council of Canada (NSERC), Discovery Horizons Program and Individual Discovery Grant

---------------------------
SHARING/ACCESS INFORMATION
---------------------------

1. Licenses/restrictions placed on the data:  These data are available under a CC BY 4.0 license <https://creativecommons.org/licenses/by/4.0/> 

2. Links to publications that cite or use the data: https://doi.org/10.1016/j.compbiomed.2025.110003
@article{DBLP:journals/cbm/ElmekkiASSZZBKXPMOSM25,
  author       = {Hanae Elmekki and
                  Ahmed Alagha and
                  Hani Sami and
                  Amanda Spilkin and
                  Antonela Zanuttini and
                  Ehsan Zakeri and
                  Jamal Bentahar and
                  Lyes Kadem and
                  Wen{-}Fang Xie and
                  Philippe Pibarot and
                  Rabeb Mizouni and
                  Hadi Otrok and
                  Shakti Singh and
                  Azzam Mourad},
  title        = {{CACTUS:} An open dataset and framework for automated Cardiac Assessment
                  and Classification of Ultrasound images using deep transfer learning},
  journal      = {Comput. Biol. Medicine},
  volume       = {190},
  pages        = {110003},
  year         = {2025},
  url          = {https://doi.org/10.1016/j.compbiomed.2025.110003},
  doi          = {10.1016/J.COMPBIOMED.2025.110003},
  timestamp    = {Sun, 06 Jul 2025 13:23:07 +0200},
  biburl       = {https://dblp.org/rec/journals/cbm/ElmekkiASSZZBKXPMOSM25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

3. Was data derived from another source? no

4. Recommended citation for this dataset: Elmekki, H., Alagha, A., Sami, H., Spilkin, A., Zanuttini, A.M., Zakeri, E., Bentahar, J., Kadem, J., Xie, W.F., Pibarot, P., Mizouni, R., Otrok, H., Singh, S., and Mourad, A. (2025). CACTUS: An open dataset and framework for automated Cardiac Assessment and Classification of Ultrasound images using deep transfer learning. Federated Research Data Repository. DOI: 10.20383/103.01484


---------------------
DATA & FOLDER OVERVIEW
---------------------

1. Folder List

   A. Folder name: Images Dataset      
      Short description: This folder contains the images used for training, validation, and testing the AI framework. It is organized into six subfolders: five representing different cardiac views and one containing random images. Each image is named starting with a number, which indicates the grade of the image. The included cardiac views are:

 	● Apical Four Chamber (A4C)  
  	● Subcostal Four Chamber (SC)  
  	● Parasternal Long Axis (PL)  
  	● Parasternal Short Axis - Aortic Valve (PSAV)  
  	● Parasternal Short Axis - Mitral Valve (PSMV)
  
   B. Folder name: Grades
      Short description: This folder contains CSV files for each cardiac view, listing the grades for each image. The grading was conducted by cardiovascular imaging experts, with random images assigned a grade of 0, and other cardiac views graded on a scale from 1 to 10.

   C. Folder name: Videos      
      Short description: This folder is subdivided into two subfolders:
	● Training: Contains videos and corresponding CSV files for the classes and grades.
	● Real-Time Scan: Contains a real-time scanning scenario with the output results through the proposed framework. The scan itself is also provided separately (without the output) for reference.


2. Relationship between folders, if important: The folder “Grades” contains the grades assigned to the images stored in the folder “Image Dataset”.

3. Additional related data collected that was not included in the current data package: No additional related data.

4. Are there multiple versions of the dataset? no


---------------------------
METHODOLOGICAL INFORMATION
---------------------------

1. Description of methods used for collection/generation of data: The data were collected by scanning the CAE Blue Phantom with the GE M4S Matrix Probe and the GE Healthcare Vivid-Q ultrasound machine.

2. Methods for processing the data: The phantom scanning process produces ultrasound images that are saved in a computer linked to the ultrasound machine. The images are categorized into the predefined cardiac views and are then stored in a repository for evaluation by skilled cardiovascular imaging experts.

3. Instrument- or software-specific information needed to interpret the data: CAE Blue Phantom with the GE M4S Matrix Probe and the GE Healthcare Vivid-Q ultrasound machine

4. Environmental/experimental conditions: to achieve optimal and clear views of the targeted structures. These parameters, including depth, gain, dynamic range, frequency and power, are fundamental for guiding a cardiac US examination. The range of values for these parameters is documented in the paper publishing this dataset.

5. Describe any quality-assurance procedures performed on the data:  The CACTUS dataset is evaluated by imaging experts, who have created a grading system centered on two key factors: completeness and clarity. Completeness evaluates the visibility of the targeted cardiac structures in the image, assigning higher grades to images displaying the entire structure compared to those revealing only partial views. Clarity examines the luminosity of images and their purity from speckles and noise. The grading scale spans from 0 to 10, where 0 signifies an image that fails to capture a specific cardiac window, rendering it uninterpretable, whereas 10 represents a fully visible cardiac view with distinctly identifiable structures and optimal gain/power settings for clear delineation. 

6. People involved with sample collection, processing, analysis and/or submission: Amanda Spilkin and Antonela Mariel Zanuttin (medical imaging experts).

-----------------------------------------------------------------
DATA-SPECIFIC INFORMATION FOR: CACTUS
-----------------------------------------------------------------

1. dataset consists of image, video, and CSV files, not tabular variables.

2. Total number of images: 37,736  
      – Apical Four Chamber (A4C): 7,422  
      – Subcostal Four Chamber (SC): 6,345  
      – Parasternal Long Axis (PL): 6,102  
      – Parasternal Short Axis – Aortic Valve (PSAV): 5,832  
      – Parasternal Short Axis – Mitral Valve (PSMV): 6,014  
      – Random Images: 6,021 

3. Missing data codes: None (all images and corresponding grade files are included)