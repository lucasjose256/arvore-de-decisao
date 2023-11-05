from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree
import pandas as pd
import pydot
import graphviz

# Suponha que você tenha um DataFrame chamado 'data' com suas colunas de dados


# Abre o arquivo para leitura
with open('vital_signs_132v.txt', 'r') as arquivo:
    # Lê as linhas do arquivo
    linhas = arquivo.readlines()

# Inicializa uma lista vazia para armazenar a matriz
matriz = []

for linha in linhas:
    # Divide a linha em elementos separados por vírgula
    elementos = linha.strip().split(',')
    # Converte os elementos em números de ponto flutuante e os adiciona à matriz
    linha_da_matriz = [float(elemento) for elemento in elementos]
    matriz.append(linha_da_matriz)

df = pd.DataFrame(matriz)
# Opcionalmente, você pode nomear as colunas
colunas = ["id", "pSist", "pDiast", "qPA", "pulso", "resp", "gravid", "classe", ]
df.columns = colunas
print(df)
# Agora você tem um DataFrame a partir da matriz
X = df[['qPA', 'pulso', 'resp']]
y = df['classe']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = tree.DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

print(clf.predict(X))
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Acurácia:", accuracy)
print("Precisão:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

dot_data = export_graphviz(clf, out_file='/Users/lucasbarszcz/Desktop/Faculdade/sistemas inteligentes/image.html',
                           feature_names=['qPA', 'pulso', 'resp'], class_names=['1.0', '2.0', '3.0', '4.0'], filled=True,
                           rounded=True, proportion=True, node_ids=True, rotate=False, label='all',
                           special_characters=True, )
# Split the lines of the DOT data
# Create a graph from the DOT data
graph = graphviz.Source(dot_data)
# View the decision tree

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
