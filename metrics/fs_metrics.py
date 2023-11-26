import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from torchmetrics import Metric


class FaceSemanticMetrics(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = InceptionResnetV1(pretrained="vggface2").eval()
        self.add_state("fsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("cos_sim", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, generated_images, real_images):
        with torch.no_grad():
            features_gen = self.model(generated_images)
            features_real = self.model(real_images)

            # Calculate FSD (Face Semantic Distance)
            fsd = torch.sum(torch.abs(features_gen - features_real), dim=1)

            # Calculate FSS (Face Semantic Similarity)
            fss = torch.cos(F.pairwise_distance(features_gen - features_real))

            # Calculate Cosine Similarity
            cos_sim = F.cosine_similarity(features_gen, features_real)

            self.fsd += fsd.sum()
            self.fss += fss.sum()
            self.cos_sim += cos_sim.sum()
            self.total += generated_images.size(0)

    def compute(self):
        avg_fsd = self.fsd.float() / self.total
        avg_fss = self.fss.float() / self.total
        avg_cos_sim = self.cos_sim.float() / self.total
        return avg_fsd, avg_fss, avg_cos_sim
