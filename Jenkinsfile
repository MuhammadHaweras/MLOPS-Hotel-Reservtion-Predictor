pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages {
        stage("Cloning Github repo to jenkins"){
            steps{
                script{
                    echo "Cloning the repository from GitHub........"
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/MuhammadHaweras/MLOPS-Hotel-Reservtion-Predictor.git']])

                }
            }
        }

        stage("Setting conda env and installing dependencies"){
            steps{
                script{
                    echo "Setting conda env and installing dependencies........"
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