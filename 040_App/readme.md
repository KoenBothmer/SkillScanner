## Launch
To run the app locally:
	- Clone this repository \n
	- Open a command shell \n
	- Navigate to ./04_app
	- Build the docker image: Run command "docker build --tag embeddings ."
	- Run the image and publish the app to port 5000 of localhost: run "docker run --publish 5000:5000 embeddings"
	- Access the app through localhost:5000 from your web-browser

To deploy the app to Google Cloud Platform: Cloud Run:
	- Make sure to have the latest Google Cloud Software Development Kit installed on your local machine
	- Open a command shell
	- Navigate to ./04_app
	- Submit the Docker image to Google Cloud: Run command "gcloud builds submit --tag gcr.io/thesis-clustering/clusterer"
	- Deploy the container as a web-app: Run command "gcloud run deploy --image gcr.io/thesis-clustering/clusterer --platform managed --port 5000 --memory 4G"

To only access our web-app:
https://skillscanner-x3gfl3cnea-ew.a.run.app/
