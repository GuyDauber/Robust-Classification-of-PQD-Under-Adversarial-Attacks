# Robust-Classification-of-PQD-Under-Adversarial-Attacks
This Project was created for the Technion as part of my Bachelor's degree in the ECE faculty.
This project contains presentation, poster and report.
The code contains MATLAB and python files for the dataset generation and processing of the dataset signals.
The MATLAB GUI creates a dataset of Power Quality Disturbances signals.
Overall, there are 16 different types of signals, including normal signal and combination of variety of disturbances.
Each signal is defined as voltage [P.U.] as function of time
The user can choose the nonimal frequency (50 or 60 Hz), number of cycles, and to add a white gaussian noise.
The PQD signals are then going through a process of Continuous Wavelet Transform method, and become scalogram images after more processing.
The images are then classified for a base line peroformance, and then beign investigated under adversarial attacks from different types and modes.
The performance is measured using accuracy, f1 score, fooling rate, and average robustness.
The last stage is a simple defensive method, adversarial training, to boost the decreased peroformance due to the attacks.
