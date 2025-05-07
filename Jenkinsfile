pipeline{
    agent any

    environment{
        VENV_DIR='venv'
        GCP_PROJECT="hotelprediction"
        GCLOUD_PATH="/var/jenkins_home/google-cloud-sdk/bin"
    }
    
    stages{
        stage('Cloning Github repo to Jenkins'){
            steps{
                script{
                    echo '################# Cloning Github ########################'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/WesleyG31/hotel-reservation-prediction.git']])
                }
            }
        }
        stage('Setting up Virtual Enviroment and Installing Dependencies'){
            steps{
                script{
                    echo '################# Env and Dependencies #######################'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }
        stage('Building and Pushing Docker Image to GCR'){
            steps{
                withCredentials([file(credentialsId : 'gcp-key', variable:'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script{
                        echo '##################### Building and Pushing Docker Image to GCR ###################'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}

                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}

                        gcloud auth configure-docker --quiet

                        docker build -t gcr.io/${GCP_PROJECT}/hotel-prediction:latest .

                        docker push gcr.io/${GCP_PROJECT}/hotel-prediction:latest

                        '''
                    }
                }
            }
        }
        stage('Deploy to Google Cloud Run'){
            steps{
                withCredentials([file(credentialsId : 'gcp-key', variable:'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script{
                        echo '##################### Deploy to Google Cloud Run ###################'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}

                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}

                        gcloud run deploy hotel-prediction \
                            --image=gcr.io/${GCP_PROJECT}/hotel-prediction:latest\
                            --platform=managed \
                            --region=us-central1 \
                            --allow-unauthenticated
                        '''
                    }
                }
            }
        }
    }
}