cd C:\Users\BothmK01\OneDrive - FrieslandCampina\Data science master\Thesis\Code\App\docker
docker build --tag embeddings .
docker run --publish 5000:5000 embeddings