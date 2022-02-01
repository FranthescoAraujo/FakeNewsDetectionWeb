import joblib
import numpy
import sklearn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from gensim.models import Word2Vec
from flask import Flask, render_template, request, flash 
from PreProcessing import PreProcessing
from CarregaPLN import CarregaPLN
from CarregarClassificador import CarregarClassificador
from flask_toastr import Toastr

app = Flask(__name__)
toastr = Toastr(app)
app.config['SECRET_KEY'] = 'Toastr'

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/", methods=['GET', 'POST'])
def getNLPAndClassifier():
    texto = request.form.get('texto')
    idioma = request.form.get('dropdownIdioma')
    removeStopWords = request.form.get('dropdownRemoveStopWords')
    PLN = request.form.get('dropdownPLN')
    representacaoVetores = request.form.get('dropdownRepresentacaoVetores')
    classificador = request.form.get('dropdownClassificador')
    camadaOculta = request.form.get('dropdownCamadaOculta')
    representacaoDocumento = request.form.get('dropdownRepresentacaoDocumento')
    if validarEntradas(texto, PLN, classificador, camadaOculta, representacaoDocumento):
        return render_template('index.html')
    texto = preProcessamento(texto, idioma, removeStopWords)
    representacao = processamentoLinguagemNatural(texto, idioma, removeStopWords, PLN, representacaoVetores, representacaoDocumento)
    resultado = classificar(representacao, idioma, removeStopWords, PLN, representacaoVetores, classificador, camadaOculta, representacaoDocumento)
    return render_template('index.html', text = "Resultado da classificação: ", result = resultado)

def preProcessamento(texto, idioma, removeStopWords):
    texto = [texto]
    texto = PreProcessing.removeAccentuation(texto)
    texto = PreProcessing.removeSpecialCharacters(texto)
    texto = PreProcessing.removeNumerals(texto)
    texto = PreProcessing.toLowerCase(texto)
    if (bool(removeStopWords)):
        texto = PreProcessing.removeStopWords(texto, idioma)
    return texto

def processamentoLinguagemNatural(texto, idioma, removeStopWords, PLN, representacaoVetores, representacaoDocumento):
    load = CarregaPLN(texto, idioma, removeStopWords, PLN, representacaoVetores, representacaoDocumento)
    representacao = []
    if (PLN == "Doc2vec - PV-DM" or PLN == "Doc2vec - PV-DBOW"):
        representacao = load.doc2vec()
    if (PLN == "Doc2vec - Concatenated"):
        representacao = load.doc2vecConcatenated()
    if (PLN == "Word2vec - Skipgram - Sum" or PLN == "Word2vec - CBOW - Sum"):
        representacao = load.word2vec(0)
    if (PLN == "Word2vec - Skipgram - Average" or PLN == "Word2vec - CBOW - Average"):
        representacao = load.word2vec(1)
    if (PLN == "Word2vec - Skipgram - Matrix" or PLN == "Word2vec - CBOW - Matrix"):
        representacao = load.word2vecMatrix()
    if (PLN == "Word2vec - Skipgram - Matrix Transposed" or PLN == "Word2vec - CBOW - Matrix Transposed"):
        representacao = load.word2vecMatrixTransposed()
    if (PLN == "Tensorflow Embedding"):   
        representacao = load.tensorflowEmbedding()    
    return representacao

def classificar(representacao, idioma, removeStopWords, PLN, representacaoVetores, classificador, camadaOculta, representacaoDocumento):
    load = CarregarClassificador(representacao, idioma, removeStopWords, PLN, representacaoVetores, classificador, camadaOculta, representacaoDocumento)
    if (classificador == "SVM"):
        resultado = load.svm()
    if (classificador == "Naive Bayes"):
        resultado = load.naiveBayes()
    if (classificador == "RNA"):
        resultado = load.rna()
    if (classificador == "LSTM"):
        resultado = load.lstm()
    if (classificador == "LSTM With Embedding"):
        resultado = load.lstmWithEmbedding()
    return resultado

def validarEntradas(texto, PLN, classificador, camadaOculta, representacaoDocumento):
    if texto == "":
        flash("Digite o seu texto", 'warning')
        return True
    if (PLN == "Word2vec - Skipgram - Matrix" or PLN == "Word2vec - CBOW - Matrix" or PLN == "Word2vec - Skipgram - Matrix Transposed" or PLN == "Word2vec - CBOW - Matrix Transposed" or PLN == "Tensorflow Embedding") and representacaoDocumento == "null":
        flash("Para " + PLN + ", digite uma quantidade de palavras para representar o documento", 'warning')
        return True
    if (PLN == "Word2vec - Skipgram - Matrix" or PLN == "Word2vec - CBOW - Matrix" or PLN == "Word2vec - Skipgram - Matrix Transposed" or PLN == "Word2vec - CBOW - Matrix Transposed" or PLN == "Tensorflow Embedding" or PLN == "RNA") and camadaOculta == "null":
        flash("Para " + PLN + ", digite uma quantidade de neurônios na camada oculta", 'warning')
        return True
    if (PLN == "Word2vec - Skipgram - Matrix" or PLN == "Word2vec - CBOW - Matrix" or PLN == "Word2vec - Skipgram - Matrix Transposed" or PLN == "Word2vec - CBOW - Matrix Transposed") and classificador != "LSTM":
        flash("Para " + PLN + ", o classificador deve ser o " + classificador, 'warning')
        return True
    if (PLN == "Tensorflow Embedding" and classificador != "LSTM With Embedding"):
        flash("Para " + PLN + ", o classificador deve ser o " + classificador, 'warning')
        return True
    if (PLN == "Doc2vec - PV-DM" or PLN == "Doc2vec - PV-DBOW" or PLN == "Doc2vec - Concatenated" or PLN == "Word2vec - Skipgram - Sum" or PLN == "Word2vec - Skipgram - Average" or PLN == "Word2vec - CBOW - Sum" or PLN == "Word2vec - CBOW - Average") and (classificador == "LSTM" or classificador == "LSTM With Embedding"):
        flash("Para " + PLN + ", o classificador não pode ser o LSTM, nem o LSTM With Embedding", 'warning')
        return True
    return False

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=80)