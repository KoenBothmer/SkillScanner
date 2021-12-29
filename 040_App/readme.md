## Launch
To run the app locally:<br>
	- Clone this repository <br>
	- Open a command shell <br>
	- Navigate to ./04_app <br>
	- Build the docker image: Run command "docker build --tag embeddings ."<br>
	- Run the image and publish the app to port 5000 of localhost: run "docker run --publish 5000:5000 embeddings"<br>
	- Access the app through localhost:5000 from your web-browser<br><br>

To deploy the app to Google Cloud Platform: Cloud Run:<br>
	- Make sure to have the latest Google Cloud Software Development Kit installed on your local machine<br>
	- Open a command shell<br>
	- Navigate to ./04_app<br>
	- Submit the Docker image to Google Cloud: Run command "gcloud builds submit --tag gcr.io/thesis-clustering/clusterer"<br>
	- Deploy the container as a web-app: Run command "gcloud run deploy --image gcr.io/thesis-clustering/clusterer --platform managed --port 5000 --memory 4G"<br><br>

To only access our web-app:<br>
https://skillscanner-x3gfl3cnea-ew.a.run.app/
