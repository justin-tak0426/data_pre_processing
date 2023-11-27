#%%
import pandas as pd
import os
from sklearn.decomposition import PCA
import librosa
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE

    

label_csv = pd.read_csv("/home/elicer/UrbanSound8K.csv")
audio_directory = '/home/elicer/total_data'

all_features = []
label_list = []


for filename in label_csv['slice_file_name']:
    audio_path = os.path.join(audio_directory, filename)
    #features = audio_to_feature_vector(audio_path)
    features = np.random.rand(6)   # 실제로 여기 feature에는 길이가 동일된 mfcc값이 들어가야 함. (1차원 값)
    all_features.append(features)
    condition = (label_csv['slice_file_name'] == filename)
    label_list.append(label_csv[condition]['classID'])

#print(label_list)

# 특징 매트릭스 생성
feature_matrix = np.vstack(all_features)

def pca_plot(dimension, feature_matrix, class_num, width, height):
    # PCA 적용
    n_components = dimension
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(feature_matrix)
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)

    cmap = px.colors.sequential.Viridis  # Viridis 색상 팔레트 사용

    if dimension == 2:
        # DataFrame 생성
        df = pd.DataFrame({'Feature1': pca_result[:, 0], 'Feature2': pca_result[:, 1], 'Class': label_list})

        # 'Class' 열의 값을 리스트로 변환
        df['Class'] = df['Class'].apply(lambda x: x.iloc[0])

        # 2D 산점도 그리기
        fig = px.scatter(df, x='Feature1', y='Feature2', size_max=1, color='Class',
                        title='PCA 2D Scatter Plot', labels={'Feature1': 'Feature 1', 'Feature2': 'Feature 2', 'Class': 'Class'})

        # 레전드 위치 조정
        fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, itemsizing='constant'))

        # 클래스별 레전드 색상 설정
        for i in range(class_num):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                    marker=dict(size=20, color=cmap[i], showscale=False),
                                    legendgroup=f'Class {i}', name=f'Class {i}'))

    elif dimension == 3:
        # DataFrame 생성
        df = pd.DataFrame({'Feature1': pca_result[:, 0], 'Feature2': pca_result[:, 1], 'Feature3': pca_result[:, 2], 'Class': label_list})
        df['Class'] = df['Class'].apply(lambda x: x.iloc[0])

        # 3D 산점도 그리기
        fig = px.scatter_3d(df, x='Feature1', y='Feature2', z='Feature3', size_max=1, color='Class',
                            title='PCA 3D Scatter Plot',
                            labels={'Feature1': 'Feature 1', 'Feature2': 'Feature 2', 'Feature3': 'Feature 3', 'Class': 'Class'})

        # 레전드 위치 조정
        fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

        # 클래스별 레전드 색상 설정
        for i in range(class_num):
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                       marker=dict(size=20, color=cmap[i], showscale=False),
                                       legendgroup=f'Class {i}', name=f'Class {i}'))

    else:
        print("Wrong dimension")
        return None

    # iplot 사용
    fig.update_layout(width=width, height=height)
    fig.show()

# 함수 호출 예시
#pca_plot(2, feature_matrix, 10, 1200, 800)  # 2차원 산점도
#pca_plot(3, feature_matrix, 10, 1000, 1000)  # 3차원 산점도


def tsne_plot(feature_matrix, class_num, width=1000, height=800):
    # t-SNE 적용
    n_components = 2
    tsne = TSNE(n_components=n_components, random_state=42)
    tsne_result = tsne.fit_transform(feature_matrix)
    
    cmap = px.colors.sequential.Viridis  # Viridis 색상 팔레트 사용

    
    # DataFrame 생성
    df = pd.DataFrame({'Feature1': tsne_result[:, 0], 'Feature2': tsne_result[:, 1], 'Class': label_list})

    # 'Class' 열의 값을 리스트로 변환
    df['Class'] = df['Class'].apply(lambda x: x.iloc[0])

    # 2D 산점도 그리기
    fig = px.scatter(df, x='Feature1', y='Feature2', size_max=5, color='Class',
                    title='t-SNE 2D Scatter Plot', labels={'Feature1': 'Feature 1', 'Feature2': 'Feature 2', 'Class': 'Class'})

    # 레전드 위치 조정
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, itemsizing='constant'))

    # 클래스별 레전드 색상 설정
    for i in range(class_num):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                marker=dict(size=5, color=cmap[i], showscale=False),
                                legendgroup=f'Class {i}', name=f'Class {i}'))

    fig.update_layout(width=width, height=height)
    fig.show()

tsne_plot(feature_matrix, 10, width=700, height=700)
# %%
