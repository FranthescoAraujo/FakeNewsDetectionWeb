import numpy
from gensim.models import Word2Vec, Doc2Vec
from keras.preprocessing import sequence

class CarregaPLN:
    def __init__(self, texto, idioma, removeStopWords, PLN, representacaoVetores, representacaoDocumento):
        self.texto = texto[0]
        self.idioma = idioma
        self.removeStopWords = removeStopWords
        self.PLN = PLN
        self.representacaoVetores = representacaoVetores
        self.representacaoDocumento = representacaoDocumento

    def createPathWithVectorSize(self):
        self.path = "Models/NaturalLanguageProcessing/" + self.idioma + "/RemoveStopWords-" + self.removeStopWords + "/" + self.PLN + "/VectorSize-" + self.representacaoVetores + "/"

    def createPathWithMatrixSize(self):
        self.path = "Models/NaturalLanguageProcessing/" + self.idioma + "/RemoveStopWords-" + self.removeStopWords + "/" + self.PLN + "/MatrixSize-" + self.representacaoDocumento + "/"

    def doc2vec(self):
        self.createPathWithVectorSize()
        model = Doc2Vec.load(self.path + "model.model")
        return model.infer_vector(self.texto.split()).tolist()

    def doc2vecConcatenated(self):
        self.createPathWithVectorSize()
        model1 = Doc2Vec.load(self.path + "model1.model")
        model2 = Doc2Vec.load(self.path + "model2.model")
        return model1.infer_vector(self.texto.split()).tolist() + model2.infer_vector(self.texto.split()).tolist()
    
    def word2vec(self, meanOrSum):
        self.createPathWithVectorSize()
        model = Word2Vec.load(self.path + "model.model").wv
        numWords = len(self.texto.split())
        documentRepresentation = [0]*int(self.representacaoVetores)
        for word in self.texto.split():
            if word not in model:
                    continue
            if meanOrSum == 0:
                documentRepresentation+=model[word]
            if meanOrSum == 1:
                documentRepresentation+=model[word]
                numWords = 1
        return numpy.divide(documentRepresentation, numWords).tolist()

    def word2vecMatrix(self):
        self.createPathWithVectorSize()
        model = Word2Vec.load(self.path + "model.model").wv
        documentRepresentation = [[0]*int(self.representacaoVetores)]*int(self.representacaoDocumento)
        index = 0
        for word in self.texto.split():
            if index >= int(self.representacaoDocumento):
                break
            if word in model:
                documentRepresentation[index] = model[word]
                index += 1
        return documentRepresentation

    def word2vecMatrixTransposed(self):
        documentRepresentation = self.word2vecMatrix()
        return numpy.transpose(documentRepresentation, (1,0))

    def tensorflowEmbedding(self):
        self.createPathWithMatrixSize()
        i = 0
        document = self.texto.split()
        while i < len(document):
            if document[i] in self.word2id:
                document[i] = self.word2id[document[i]]
                i += 1
                continue
            del document[i]
        return sequence.pad_sequences(document, maxlen=self.representacaoDocumento, padding='post')
