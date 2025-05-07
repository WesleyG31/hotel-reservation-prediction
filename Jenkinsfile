pipeline{
    agent any
    
    stages{
        stage('Cloning Github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning Github...'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/WesleyG31/hotel-reservation-prediction.git']])
                }
            }
        }
    }
}