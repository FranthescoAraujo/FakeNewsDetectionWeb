import joblib
import tensorflow.keras as keras

class CarregarClassificador:
    def __init__(self, representacao, idioma, removeStopWords, PLN, representacaoVetores, classificador, camadaOculta, representacaoDocumento):
        self.representacao = [representacao]
        self.idioma = idioma
        self.removeStopWords = removeStopWords
        self.PLN = PLN
        self.representacaoVetores = representacaoVetores
        self.classificador = classificador
        self.camadaOculta = camadaOculta
        self.representacaoDocumento = representacaoDocumento

    def svm(self):
        path = "Models/Classifiers/" + self.classificador + "/" + self.idioma + "/RemoveStopWords-" + self.removeStopWords + "/" + self.PLN + "/VectorSize-" + self.representacaoVetores + "/"
        classificador = joblib.load(path + "classifier.joblib.pkl")
        return self.returnFakeOrTrue(classificador.predict(self.representacao))

    def naiveBayes(self):
        path = "Models/Classifiers/" + self.classificador + "/" + self.idioma + "/RemoveStopWords-" + self.removeStopWords + "/" + self.PLN + "/VectorSize-" + self.representacaoVetores + "/"
        classificador = joblib.load(path + "classifier.joblib.pkl")
        return self.returnFakeOrTrue(classificador.predict(self.representacao))

    def rna(self):
        path = "Models/Classifiers/" + self.classificador + "/ClassifierSize-" + self.camadaOculta + "/" + self.idioma + "/RemoveStopWords-" + self.removeStopWords + "/" + self.PLN + "/VectorSize-" + self.representacaoVetores + "/"
        classificador = keras.models.load_model(path + "classifier")
        return self.returnFakeOrTrue(classificador.predict(self.representacao))

    def lstm(self):
        path = "Models/Classifiers/" + self.classificador + "/ClassifierSize-" + self.camadaOculta + "/MatrixSize-" + self.representacaoDocumento + "/" + self.idioma + "/RemoveStopWords-" + self.removeStopWords + "/" + self.PLN + "/VectorSize-" + self.representacaoVetores + "/"
        classificador = keras.models.load_model(path + "classifier")
        return self.returnFakeOrTrue(classificador.predict(self.representacao))
    
    def lstmWithEmbedding(self):
        path = "Models/Classifiers/" + self.classificador + "/ClassifierSize-" + self.camadaOculta + "/VectorSize-" + self.representacaoVetores + "/" + self.idioma + "/RemoveStopWords-" + self.removeStopWords + "/" + self.PLN + "/MatrixSize-" + self.representacaoDocumento + "/"
        classificador = keras.models.load_model(path + "classifier")
        return self.returnFakeOrTrue(classificador.predict(self.representacao))

    def returnFakeOrTrue(self, resultado):
        if (resultado[0] == 1):
            return "Fake"
        return "True"