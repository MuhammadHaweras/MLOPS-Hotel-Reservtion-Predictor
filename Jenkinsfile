pipeline{
    agent any

    stages {
        stage("Cloning Github repo to jenkins"){
            steps{
                script{
                    echo "Cloning the repository from GitHub........"
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/MuhammadHaweras/MLOPS-Hotel-Reservtion-Predictor.git']])
                    
                }
            }
        }
    }
}