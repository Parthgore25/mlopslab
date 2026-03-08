pipeline {
    agent any
    
    environment {
        PYTHON_ENV = 'ml-training-env'
    }
    
    parameters {
        choice(name: 'ML_FRAMEWORK', 
               choices: ['sklearn', 'tensorflow', 'pytorch'],
               description: 'Select ML framework for training')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                // Changed 'sh' to 'bat' and used Windows pathing
                bat """
                    python -m venv ${PYTHON_ENV}
                    call ${PYTHON_ENV}\\Scripts\\activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                """
            }
        }
        
        stage('Data Validation') {
            steps {
                bat """
                    call ${PYTHON_ENV}\\Scripts\\activate
                    python -c "import pandas as pd; df = pd.read_csv('data/dataset.csv'); print(f'Data shape: {df.shape}')"
                """
            }
        }
        
        stage('Train Model') {
            steps {
                script {
                    def trainScript = "src/train_${params.ML_FRAMEWORK}.py"
                    bat """
                        call ${PYTHON_ENV}\\Scripts\\activate
                        python ${trainScript}
                    """
                }
            }
        }
        
        stage('Archive Artifacts') {
            steps {
                // This preserves the model files for your lab report [cite: 507]
                archiveArtifacts artifacts: 'models/**/*', fingerprint: true
            }
        }
    }
    
    post {
        success { echo 'Training completed successfully!' }
        failure { echo 'Training failed!' }
        always { cleanWs() }
    }
}