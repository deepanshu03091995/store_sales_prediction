# store_sales_prediction

# Deployed_link - http://20.193.148.154/
### Problem Statement
Nowadays, shopping malls and Big Marts keep track of individual item sales data in
order to forecast future client demand and adjust inventory management. In a data
warehouse, these data stores hold a significant amount of consumer information and
particular item details. By mining the data store from the data warehouse, more
anomalies and common patterns can be discovered.
## Tech Stack Used

1. Python
2. FastAPI
3. Machine learning algorithms
4. Docker
5. MongoDB

## Infrastructure Required.

1. AWS S3
2. AWS EC2
3. AWS ECR
4. Git Actions
5. Terraform

## How to run?

Before we run the project, make sure that you are having MongoDB in your local system, with Compass since we are using MongoDB for data storage. You also need AWS account to access the service like S3, ECR and EC2 instances.

## Data Collections

![image](https://user-images.githubusercontent.com/57321948/193536736-5ccff349-d1fb-486e-b920-02ad7974d089.png)

## Project Archietecture

![image](https://user-images.githubusercontent.com/57321948/193536768-ae704adc-32d9-4c6c-b234-79c152f756c5.png)

## Deployment Archietecture

![image](https://user-images.githubusercontent.com/57321948/193536973-4530fe7d-5509-4609-bfd2-cd702fc82423.png)

### Step 1: Clone the repository

```bash
git clone https://github.com/deepanshu03091995/store_sales_prediction.git
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -n sales python=3.8.6 -y
```

```bash
conda activate sales
```

### Step 3 - Install the requirements

```bash
pip install -r requirements.txt
```

### Step 4 - Export the environment variable

```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>

export MONGODB_URL="mongodb+srv://<username>:<password>@ineuron-ai-projects.7eh1w4s.mongodb.net/?retryWrites=true&w=majority"

```

### Step 5 - Run the application server

```bash
python app.py
```

### Step 6. Train application

```bash
http://localhost:8080/train

```

### Step 7. Prediction application

```bash
http://localhost:8080/predict

```

## Run locally

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image

```
docker build --build-arg AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> --build-arg AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> --build-arg AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION> --build-arg MONGODB_URL=<MONGODB_URL> .

```

3. Run the Docker image

```
docker run -d -p 8080:8080 <IMAGE_NAME>
```

To run the project first execute the below commmand.
MONGO DB URL:

```
mongodb+srv://avnish:XglZZ9OkjjUw74pZ@ineuron-ai-projects.7eh1w4s.mongodb.net/admin?authSource=admin&replicaSet=atlas-okvkrd-shard-0&w=majority&readPreference=primary&appname=MongoDB%20Compass&retryWrites=true&ssl=true
```

windows user

```
MONGO_DB_URL=mongodb+srv://avnish:XglZZ9OkjjUw74pZ@ineuron-ai-projects.7eh1w4s.mongodb.net/admin?authSource=admin&replicaSet=atlas-okvkrd-shard-0&w=majority&readPreference=primary&appname=MongoDB%20Compass&retryWrites=true&ssl=true
```

Linux user

```
export MONGO_DB_URL=mongodb+srv://avnish:XglZZ9OkjjUw74pZ@ineuron-ai-projects.7eh1w4s.mongodb.net/admin?authSource=admin&replicaSet=atlas-okvkrd-shard-0&w=majority&readPreference=primary&appname=MongoDB%20Compass&retryWrites=true&ssl=true
```

then run

```
python main.py
```
