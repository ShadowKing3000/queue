from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


app = FastAPI()

# GNN model class
class GNNRoutingModel(torch.nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super(GNNRoutingModel, self).__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_predictor = torch.nn.Linear(2*hidden_dim + edge_feat_dim, 1)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        src, dst = edge_index
        edge_inputs = torch.cat([x[src], x[dst], edge_attr], dim=1)
        edge_scores = self.edge_predictor(edge_inputs)
        return edge_scores

# Load GNN model
try:
    node_feat_dim = 5
    edge_feat_dim = 3
    hidden_dim = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNRoutingModel(node_feat_dim, edge_feat_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load("gnn_model.pth"))
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Request schema
class GraphRequest(BaseModel):
    counter_id_list: list
    queue_length_list: list
    status_list: list
    processing_rate_list: list
    delay_list: list
    capacity_list: list
    neighbor_counter_id_list: list
    distance_list: list
    reroute_capacity_list: list
    is_connected_list: list
    start_counter: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the GNN Routing API"}

@app.post("/predict_route")
def predict_route(request: GraphRequest):
    try:
        data = {
            'counter_id': request.counter_id_list,
            'queue_length': request.queue_length_list,
            'status': request.status_list,
            'processing_rate': request.processing_rate_list,
            'delay': request.delay_list,
            'capacity': request.capacity_list,
            'neighbor_counter_id': request.neighbor_counter_id_list,
            'distance': request.distance_list,
            'reroute_capacity': request.reroute_capacity_list,
            'is_connected': request.is_connected_list
        }
        df = pd.DataFrame(data)

        le = LabelEncoder()
        all_counters = list(set(request.counter_id_list + request.neighbor_counter_id_list))
        le.fit(all_counters)

        df['counter_id_encoded'] = le.transform(df['counter_id'])
        df['neighbor_counter_id_encoded'] = le.transform(df['neighbor_counter_id'])

        node_list = pd.DataFrame({'counter_id': all_counters})
        node_list['counter_id_encoded'] = le.transform(node_list['counter_id'])

        node_features = pd.merge(node_list, df[['counter_id', 'queue_length', 'status', 'processing_rate', 'delay', 'capacity']],
                                 on='counter_id', how='left')
        node_features['queue_length'] = node_features['queue_length'].fillna(0)
        node_features['status'] = node_features['status'].fillna(1)
        node_features['processing_rate'] = node_features['processing_rate'].fillna(5.0)
        node_features['delay'] = node_features['delay'].fillna(0)
        node_features['capacity'] = node_features['capacity'].fillna(50)

        scaler = StandardScaler()
        x_node = torch.tensor(scaler.fit_transform(node_features.drop(columns=['counter_id', 'counter_id_encoded'])), dtype=torch.float)

        edge_index = torch.tensor(df[['counter_id_encoded', 'neighbor_counter_id_encoded']].values.T, dtype=torch.long)
        edge_features = torch.tensor(df[['distance', 'reroute_capacity', 'is_connected']].values, dtype=torch.float)

        graph_data = Data(x=x_node, edge_index=edge_index, edge_attr=edge_features)
        graph_data = graph_data.to(device)

        with torch.no_grad():
            predicted_delays = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr).cpu().numpy()

        G = nx.DiGraph()
        for idx in range(graph_data.edge_index.shape[1]):
            src = le.inverse_transform([graph_data.edge_index[0, idx].item()])[0]
            dst = le.inverse_transform([graph_data.edge_index[1, idx].item()])[0]
            G.add_edge(src, dst, weight=predicted_delays[idx][0])

        if request.start_counter not in G.nodes:
            raise HTTPException(status_code=404, detail="Start counter not found in graph.")

        possible_paths = {}
        for target in list(G.nodes):
            if request.start_counter != target:
                try:
                    path = nx.shortest_path(G, source=request.start_counter, target=target, weight='weight')
                    path_delay = sum(G[u][v]['weight'] for u,v in zip(path[:-1], path[1:]))
                    possible_paths[target] = {
                        "path": path,
                        "total_predicted_delay_minutes": round(path_delay, 2)
                    }
                except nx.NetworkXNoPath:
                    continue

        if not possible_paths:
            return {"message": "No paths found from the start counter."}

        sorted_paths = dict(sorted(possible_paths.items(), key=lambda x: x[1]['total_predicted_delay_minutes']))

        return {
            "start_counter": request.start_counter,
            "reroute_options": sorted_paths
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
