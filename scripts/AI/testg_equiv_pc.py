from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
from src.trainers.g_equiv_trainer import SO3EquivTrainer
from src.models.Equiv.simple_decoder import FoldingDecoder, SimpleFreqGridDecoder
from src.models.Equiv.unsupervised_g_encoder import SimpleEncoder

from src.models.Equiv.data import make_graphs_from_vertices, get_vertices_from_graph, unbatch_graphs
from src.models.Equiv.data_transforms import transform_graphs_explicit, get_y_rot
from src.models.Equiv.vtk import save_graphs_as_vtp, save_single_graph_as_vtp
from src.geometry.vtk import load_vtp_mesh, get_vtp_vertices
from src.utils.utils  import pad_vertices, recover_original_list
import glob
import os
from dotenv import load_dotenv
load_dotenv()

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)



PROJECT_ROOT = os.environ.get("PROJECT_ROOT")

vertices = []
for part in ["mouth", "nose"]:
    DATASET_1 = os.path.join(PROJECT_ROOT, "Dataset_faceparts_small", part, "*.vtp")
    vtp_files = glob.glob(DATASET_1)
    print("getting vtp_files: ", vtp_files)
    for file in vtp_files:
        data = load_vtp_mesh(file)
        verts = get_vtp_vertices(data)
        vertices.append(verts)

master_key = jax.random.PRNGKey(42)
batched_graphs = make_graphs_from_vertices(vertices, 
                                         master_key,
                                        r_max= 0.4, 
                                        dropout_rate= 0.0,
                                        noise_std = 0.0)

print(batched_graphs.n_edge)
unbatched_graphs = unbatch_graphs(batched_graphs)

for i, graph in enumerate(unbatched_graphs):
    save_single_graph_as_vtp(graph, f"init_graphs/graph_test_{i}.vtp")

step_key, master_key = jax.random.split(master_key)
n_graphs = batched_graphs.n_node.shape[0]
k1, k2 = jax.random.split(step_key)
thetas = jax.random.uniform(k2, (n_graphs,), minval=0.0, maxval=2 * jnp.pi)
rot_mats = jax.vmap(get_y_rot)(thetas)
trans_vecs = jax.random.uniform(k1, (n_graphs, 3), minval=-0.0, maxval=0.0)

shape_graphs_aug = transform_graphs_explicit(master_key, batched_graphs, rot_mats, trans_vecs, permute=False, r_max= 0.1, )
unbatched_graphs = unbatch_graphs(shape_graphs_aug)
for i, graph in enumerate(unbatched_graphs):
    save_single_graph_as_vtp(graph, f"init_graphs/aug_test_{i}.vtp")



n_true_lst = [vertices[i].shape[0] for i in range(len(vertices))]
flat_true_verts = jnp.concatenate(vertices, axis=0)


num_vertices_per_sample = shape_graphs_aug.n_node
print("-------Pretraining: ")
print("num_vertices_per_sample: ", num_vertices_per_sample)
print("n_true_lst: ", n_true_lst)
print("flat_true_verts: ", flat_true_verts.shape)

print("-------Intraining: ")
# each node has features: 

# 1 mean curvature scalar (l=0, even)
# gaussian curvature (l=0, even), 
# distance to centroid of its N neighbours (l=0, even), helps identify points on highly curved features vs. flat regions.
# local density (l=0, even)
# rel distance vectors (l=1, odd) to N=10 neighbours, must be order-invariant or sorted by some physical metric, i.e. sort by abs distance or projected distance, or attention
# 1 normal vector (l=1, odd)
# principal curvature directions (l=1, odd) 
# local shape index (l=0, even), derived from principal curvatures
# curvedness (l=0, even), derived from principal curvatures, complements curvature by quantifying the "magnitude" of bending
# gradient of curvature (l=1, odd)
# viewpoint vector, vector pointing from the sensor to the point
# local normal dispersion, highlights areas of high noise or sharp edges where surface orientation is ill-defined
# Tangent Projection of Relative Positions: Projecting  relative distance vectors into the tangent plane defined by the surface normal ($\mathbf{n}$), 
# decomposes the neighbor relations into "in-plane" and "normal-to-plane" components

encoder = SimpleEncoder(verbose = False)
decoder =  FoldingDecoder(num_samples = 256)
gt_vertices, mask = pad_vertices(vertices)


trainer = SO3EquivTrainer(encoder,decoder, learning_rate=1e-3, log_dir = "log")
final_state, final_preds = trainer.fit(
    graphs_batch = batched_graphs,
    true_verts = gt_vertices,
    padding_mask = mask,
    num_steps=10000,
    log_every = 1, 
    plot_every = 10)