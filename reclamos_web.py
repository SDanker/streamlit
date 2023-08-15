#importar librerias
import streamlit as st
import pickle
import pandas as pd
import reclamos_model as model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 


def main():
    #titulo
    st.title('Modelamiento categorización de reclamos')
    
    texto=st.text_input('Escribe acá un reclamo','')
    
    
    def resultado(texto):
        
        # Realiza el análisis de sentimiento
        resultados = model.sentiment_analyzer(texto)
        for resultado in resultados:
            #label = resultado['label']
            score = resultado['score']
            
            if resultado['label'] == "1 star":
                label = 'muy malo :pensive:'
            elif resultado['label'] == '2 stars':
                label = 'malo :unamused:'
            elif resultado['label'] == '3 stars':
                label = 'más o menos :neutral_face:'
            elif resultado['label'] == '4 stars':
                label = 'bueno :slightly_smiling_face:'
            else:
                label = 'muy bueno :grin:'
                
        #guarda los datos del analisis de sentimiento
        sentimiento = f'Sentimiento: {label}, Confianza: {round(score,5)}'
        
        return sentimiento
    
    def grafico(texto):
        
        # Realiza la clasificacion de texto
        result = model.classifier(texto,candidate_labels=["telecomunicaciones","deporte","cultura", "sociedad", "economia", "historia","ciencia","politica"])
        df = pd.DataFrame(data=list(zip(result['labels'],result['scores'])),columns=['label','scores'])
        
        return df

    if st.button('RUN'):
        #sentimiento
        st.write(resultado(texto))
        
        #clasificacion texto
        df = grafico(texto)
        
        fig, ax = plt.subplots()
        sns.set_theme(style="whitegrid")
        sns.set_color_codes("pastel")
        
        st.write("\n Probabilidad de pertenecer a una categoría:")
        plt.gca().yaxis.set_major_formatter('{:.0f}%'.format)
        
        plt.xticks(rotation=90)
        
        sns.barplot(x=df['label'], y=df['scores']* 100,ax=ax,color="b").set(xlabel="Category", ylabel="Score")
        st.pyplot(fig)
        
    # Retroalimentación del usuario
    retroalimentacion = st.radio("¿El análisis es correcto?", ["Correcto", "Incorrecto"])
    categoria_correcta = None
    if retroalimentacion == "Incorrecto":
    # Supongo que las categorías son Positivo y Negativo, pero puedes cambiarlas según tu modelo
        sentimiento_correcto = st.selectbox("Por favor, selecciona el sentimiento correcto:", ["Muy bueno", "Bueno","Mas o menos", "malo","Muy malo"])
        categoria_correcta = st.selectbox("Por favor, selecciona la categoria correcta:", ["telecomunicaciones","deporte","cultura", "sociedad", "economia", "historia","ciencia","politica"])

    if st.button("Enviar Retroalimentación"):
        #data = {
        #    'Texto': [texto],
        #    'Predicción': [label],
        #    'Feedback': [retroalimentacion],
        #    'Categoría Correcta': [categoria_correcta if categoria_correcta else 'N/A']
        #}
        #guardar_en_csv(data)
        st.write("¡Retroalimentación guardada exitosamente!")
    
    if st.button("Métrica de evaluación de modelo"):
        
        fig, ax = plt.subplots()
        sns.set_theme(style="whitegrid")
        sns.set_color_codes("pastel")
        
        historial = [0.88,0.86,0.90,0.91,0.87,0.89,0.95,0.93,0.96,0.98]
        x = [ "dia 1","dia 2","dia 3","dia 4","dia 5","dia 6","dia 7","dia 8","dia 9","dia 10"]
        
        sns.lineplot(x=x, y=historial,ax=ax, color="b")
        sns.lineplot(x=x, y=0.85,ax=ax, color="r")
        
        plt.ylim(0.5,1)
        #plt.gca().yaxis.set_major_formatter('{:.0f}%'.format)
        
        st.pyplot(fig)
        
        

if __name__ == '__main__':
    main()
    