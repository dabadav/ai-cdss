Patient Data
============

Clinician enters the patient details (:class:`ai_cdss.models.SessionSchema`) through the Medical Information Management System (MIMS), which serves as prior information of the patient.

Included information:

- **Stroke-related condition**: the paretic side (e.g., Left), the upper extremity to train (e.g., Left Arm), and the patient’s hand-raising capacity (e.g., Partial). Cognitive function level (e.g., Moderate Impairment), and whether the patient has hemineglect (e.g., Yes), both of which inform cognitive rehabilitation needs. 
- **Demographics**: gender (e.g., Female), skin color (e.g., Light Brown), and age (e.g., 72), which can influence treatment planning and outcomes.
- **Digital literacy**: patient’s experience with video games (e.g., 0, indicating none) and computer usage (e.g., 2, indicating moderate familiarity), which help determine the suitability of digital therapy interfaces. Clinicians can include comments (e.g., “Patient fatigues quickly; prefers afternoon sessions”) to record relevant observations. Finally, physical measures such as height (e.g., 165 cm) and arm size (e.g., 32 cm) ensure proper setup and adjustment of rehabilitation equipment.
- **Clinical observations and physical measures**: comments (e.g., “Patient fatigues quickly; prefers afternoon sessions”), height (e.g., 165 cm), and arm size (e.g., 32 cm) to support tailored equipment setup and session planning.

At the first clinical session, the therapist enters the patient’s baseline data into the system. If a clinical assessment is performed, standardized scales such as the ARAT (for motor function) and MoCA (for cognitive status) are also provided.