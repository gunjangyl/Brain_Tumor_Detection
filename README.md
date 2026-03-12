# Brain Tumor Detection | Web App Demo (Flask) | Team Delta
Brain Tumor Detection using Web App (Flask) that can classify if patient has brain tumor or not based on uploaded MRI image.

The image data that was used for this project is Brain MRI images for Brain tumor detaction.(https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

## Video Demo
Click on image to play :point_down:

[![Brain Tumor Detection | Web App Demo (Flask) | Team Delta](https://img.youtube.com/vi/8lfO3SjmXmM/0.jpg)](https://www.youtube.com/watch?v=8lfO3SjmXmM)



## How to Run the Project

- **Step 1: Open Terminal**
  Open the terminal or command prompt in the project root directory.

- **Step 2: Create Virtual Environment**
  ```bash
  # Windows
  py -m venv env

  # Mac/Linux
  python3 -m venv env
  ```

- **Step 3: Activate Virtual Environment**
  ```bash
  # Windows (PowerShell)
  .\env\Scripts\activate

  # Windows (Command Prompt)
  env\Scripts\activate

  # Mac/Linux
  source env/bin/activate
  ```

- **Step 4: Install Requirements**
  ```bash
  pip install -r requirements.txt
  ```

- **Step 5: Run the Flask Backend**
  ```bash
  flask run
  ```
  Check the API status at: http://127.0.0.1:5000/

## Running the React Dashboard
The frontend is located in the `frontend` directory.

- **Step 6: Open a new terminal** window.
- **Step 7: Navigate to the frontend directory:** `cd frontend`
- **Step 8: Install dependencies (first time only):** `npm install`
- **Step 9: Start the dashboard:** `npm start`
- **Step 10: Access the UI at:** http://localhost:3000
