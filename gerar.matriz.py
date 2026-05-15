import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Defina as categorias de emoções (na mesma ordem da sua imagem)
emocoes = ['Raiva', 'Nojo', 'Medo', 'Felicidade', 'Neutro', 'Tristeza', 'Surpresa']

# 2. Dados de Exemplo (Substitua pelos arrays reais da sua aplicação)
# y_real = O que o usuário realmente estava sentindo
# y_previsto = O que o seu modelo de CV detectou
y_real =     ['Neutro', 'Neutro', 'Tristeza', 'Felicidade', 'Neutro', 'Tristeza', 'Nojo', 'Neutro']
y_previsto = ['Neutro', 'Tristeza', 'Tristeza', 'Felicidade', 'Neutro', 'Neutro',   'Nojo', 'Tristeza']

# 3. Gerando a Matriz de Confusão
cm = confusion_matrix(y_real, y_previsto, labels=emocoes)

# 4. Configurando o visual do gráfico (Heatmap em tons de azul)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=emocoes, yticklabels=emocoes,
            cbar=True, square=True)

# 5. Adicionando os textos para facilitar a leitura dos erros
plt.title('Matriz de Confusão - Monitoramento Emocional no Teletrabalho', pad=20, fontsize=14)
plt.ylabel('Emoção Real (Expressa)', fontsize=12)
plt.xlabel('Emoção Prevista (Percebida pelo Modelo)', fontsize=12)

# Rotacionando os labels do eixo X para não encavalarem
plt.xticks(rotation=45) 
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()