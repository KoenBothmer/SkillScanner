## Code supporting my master Thesis thesis "Artificial Intelligence Driven Methods for the Analysis of Job Postings"

# Please Note:
For some reason, Github sometimes randomly fails to render the notebooks as provided. The notebooks can be viewed without having to download it to your own Jupyter instance through nbviewer: https://nbviewer.jupyter.org/github/KoenBothmer/thesis/.

# Main Idea:
We use artificial intelligence methods to extract and compare skills from job postings, CVs and learning curricula.
  - We evaluated several methods to achieve this goal, these are described in folders 010, 020 and 030.
  - We implemented our best method in our application Skill Scanner, which is avaiable in folder 040.
  - We conducted a user study to evaluate our app, the results are avaialbe in folder 050.
  - Along the way we conducted several experiments which are archived in folder A_miscellaneous.

## Launch
The repository has different uses, for each use the provided software should be used in an appropriate way:

# Viewing of conducted experiments
In order to only view the notebooks that describe our experiments there is no need to clone this repo. The content of all notebooks can be viewed through Github but we recommend viewing them through nbviewer: https://nbviewer.jupyter.org/github/KoenBothmer/thesis/.

# Reproducing results of conducted experiments
To guarantee identical dependencies it is reccomended to run this code from the provided Docker container.

Set-up:
- Make sure to have Git, Docker and Chrome installed on your local machine
- Clone this repository to a folder referred to as "\yourdir"
- Run "launch.bat" which will set this repository as 'origin', pull any updates to the master branch, compose the required docker container, launch Chrome to the exposed port of the Jupyter notebook (localhost:8889)
- Please note: if Chrome is not installed or not added to command prompt path it will not launch, you can navigate to localhost:8889 using a browser of your choice
- The token is "Thesis"

# Accessing our version of Skill Scanner
Our web app is available at https://skillscanner-x3gfl3cnea-ew.a.run.app/.

# Running Skill Scanner on your local machine
- Clone this repository
- Open a command shell
- Navigate to ./04_app
- Build the docker image: Run command "docker build --tag embeddings ."
- Run the image and publish the app to port 5000 of localhost: run "docker run --publish 5000:5000 embeddings"
- Access the app through localhost:5000 from your web-browser

# To deploy the app to Google Cloud Platform: Cloud Run:
- Make sure to have the latest Google Cloud Software Development Kit installed on your local machine
- Open a command shell
- Navigate to ./04_app
- Submit the Docker image to Google Cloud: Run command "gcloud builds submit --tag gcr.io/thesis-clustering/clusterer"
- Deploy the container as a web-app: Run command "gcloud run deploy --image gcr.io/thesis-clustering/clusterer --platform managed --port 5000 --memory 4G"
