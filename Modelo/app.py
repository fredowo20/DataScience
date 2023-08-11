import pandas as pd
from surprise import Reader, Dataset, SVD
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('DatasetMergeFinalCortado(1M).csv')

anime_titles = df[['anime_id', 'title', 'image_url', 'genre', 'synopsis']].drop_duplicates()

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df[['username', 'anime_id', 'my_score']], reader)
svd = SVD()

trainset = data.build_full_trainset()
svd.fit(trainset)

df_grouped = df.groupby(['anime_id', 'genre', 'score'], as_index=False)['my_score'].mean()

df_grouped = df_grouped.dropna().reset_index(drop=True)
df_grouped['genre'] = df_grouped['genre'].apply(lambda x: x.split(','))  # Asumiendo que los géneros están separados por comas

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df_grouped['genre'])

nn = NearestNeighbors(metric='jaccard')
nn.fit(genre_matrix)


def get_recommendations(anime_title):
     # Encuentra el anime_id correspondiente al título ingresado
    anime_id = anime_titles[anime_titles['title'].str.lower() == anime_title.lower()]['anime_id'].values[0]

    # Convierte anime_id en un índice
    idx = df_grouped[df_grouped['anime_id'] == anime_id].index[0]

    distances, nearest_indices = nn.kneighbors(genre_matrix[idx].reshape(1, -1), n_neighbors=20)
    # Recomendaciones con anime_id
    recommendations = df_grouped.iloc[nearest_indices[0]][['anime_id', 'score']]

    # Calcula la puntuación predicha por SVD para cada anime recomendado
    recommendations['svd_score'] = recommendations['anime_id'].apply(lambda x: svd.predict('username_example', x).est)

    # Pondera la puntuación SVD por la inversa de la distancia de género
    recommendations['hybrid_score'] = (recommendations['svd_score'] / (1 + distances[0])) * recommendations['score']

    # Ordena por la puntuación híbrida
    recommendations = recommendations.sort_values(by='hybrid_score', ascending=False)

    # Agregar el título, imagen y géneros a las recomendaciones utilizando el DataFrame de mapeo
    recommendations = recommendations.merge(anime_titles, on='anime_id', how='left')

    return recommendations

def get_anime_info(anime_id):
    return anime_titles[anime_titles['anime_id'] == anime_id].to_dict('records')[0]

# anime_id = 170
# print(get_recommendations(anime_id))

from flask import Flask, request, render_template, session, redirect, url_for, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    anime_title  = request.form['anime']
    
    # Usa la función get_recommendations para obtener los identificadores de los animes recomendados
    recommendations = get_recommendations(anime_title)['anime_id'].tolist()
    
    # Guarda los identificadores de las recomendaciones en la sesión y devuelve solo la primera
    session['recommendations'] = recommendations
    first_recommendation_id = session['recommendations'].pop(0)
    session.modified = True

    # Busca la información del primer anime recomendado en la base de datos
    first_recommendation = get_anime_info(first_recommendation_id)

    return render_template('recommend.html', rec=first_recommendation)

@app.route('/next', methods=['GET'])
def next():
    # Obtiene el identificador de la siguiente recomendación de la sesión y busca su información en la base de datos
    if session['recommendations']:
        next_recommendation_id = session['recommendations'].pop(0)
        session.modified = True
        next_recommendation = get_anime_info(next_recommendation_id)
        return render_template('recommend.html', rec=next_recommendation)
    return redirect(url_for('home'))  # Redirige a la página principal

@app.route('/search')
def search():
    term = request.args.get('term').lower()
    
    # Busca en tu DataFrame los títulos de anime que contienen 'term' (insensible a mayúsculas y minúsculas)
    matching_titles = anime_titles[anime_titles['title'].str.lower().str.contains(term)]['title'].head(15).tolist()

    return jsonify(matching_titles)  # Devuelve los títulos coincidentes como una lista JSON


if __name__ == '__main__':
    app.secret_key = 'your_secret_key'  # Necesitas configurar una clave secreta para usar sesiones
    app.run(debug=True)