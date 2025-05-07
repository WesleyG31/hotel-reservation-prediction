pipeline{
    agent any

    environment{
        VENV_DIR='venv'
    }
    
    stages{
        stage('Cloning Github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning Github...'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/WesleyG31/hotel-reservation-prediction.git']])
                }
            }
        }
        stage('Setting up Virtual Enviroment and Installing Dependencies'){
            steps{
                script{
                    echo 'Env and Dependencies........'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }
    }
}