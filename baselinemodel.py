import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.covariance
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GCN_Classifier(nn.Module):
    """
    Feature-based classifier for molecular representations.
    Works with both foundation model features and raw graph features.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5, use_features=True, multitask=False):
        super(GCN_Classifier, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_features = use_features
        self.multitask = multitask
        
        # Build the network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.feature_extractor = nn.Sequential(*layers[:-1])  # Exclude final output layer
        
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for layer in self.network:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def forward(self, features):
        """
        Forward pass for a batch of feature vectors.
        
        Args:
            features: Input feature tensor [batch_size, input_dim]
            
        Returns:
            logits: Output logits [batch_size, output_dim]
        """
        return self.network(features)
    
    def get_features(self, features):
        """
        Extract intermediate feature representations.
        
        Args:
            features: Input feature tensor [batch_size, input_dim]
            
        Returns:
            features: Feature representations [batch_size, hidden_dim]
        """
        return self.feature_extractor(features)


class BaselineOODModel(nn.Module):
    """
    Unified wrapper for all baseline OOD detection methods.
    Takes a trained classifier and applies various OOD scoring techniques.
    """
    
    def __init__(self, trained_model, method_name, args=None):
        super(BaselineOODModel, self).__init__()
        
        self.model = trained_model
        self.method_name = method_name.lower()
        self.args = args
        
        # For Mahalanobis method
        self.class_means = None
        self.precision_matrix = None
        self.num_classes = None

        # For KNN and LOF methods
        self.train_features = None
        self.knn_model = None
        self.lof_model = None
        # Feature normalization stats for distance/density methods
        self._feat_mean = None
        self._feat_std = None

        # For Conformal Prediction method
        self.calibration_scores = None

        # Set model to evaluation mode
        self.model.eval()
    
    def reset_parameters(self):
        """Reset parameters if needed."""
        pass
    
    def detect(self, features, device):
        """
        Compute OOD scores using the specified method.
        
        Args:
            features: Input feature tensor [batch_size, input_dim]
            device: torch device
            
        Returns:
            scores: OOD scores [batch_size] (higher = more likely OOD)
        """
        features = features.to(device)
        
        if self.method_name == 'msp':
            return self._msp_score(features)
        elif self.method_name == 'odin':
            return self._odin_score(features, device)
        elif self.method_name == 'energy':
            return self._energy_score(features)
        elif self.method_name == 'mahalanobis':
            return self._mahalanobis_score(features, device)
        elif self.method_name == 'knn':
            # Get k from args, default to 50
            k = getattr(self.args, 'knn_k', 50) if self.args else 50
            return self._knn_score(features, k=k)
        elif self.method_name == 'lof':
            # Get n_neighbors from args, default to 20
            n_neighbors = getattr(self.args, 'lof_neighbors', 20) if self.args else 20
            return self._lof_score(features, n_neighbors=n_neighbors)
        elif self.method_name == 'mc_dropout':
            # Get MC Dropout parameters from args
            n_samples = getattr(self.args, 'mc_dropout_samples', 20) if self.args else 20
            metric = getattr(self.args, 'mc_dropout_metric', 'entropy') if self.args else 'entropy'
            return self._mc_dropout_score(features, n_samples=n_samples, metric=metric)
        elif self.method_name == 'conformal':
            return self._conformal_score(features)
        else:
            raise ValueError(f"Unknown baseline method: {self.method_name}")
    
    def _msp_score(self, features):
        """Maximum Softmax Probability baseline - adapted for multi-task."""
        with torch.no_grad():
            logits = self.model(features)

            # Check if this is multi-task (logits shape: [batch_size, num_tasks])
            if logits.shape[1] > 2:  # Likely multi-task (more than 2 outputs)
                # For multi-task: use sigmoid probabilities instead of softmax
                sigmoid_probs = torch.sigmoid(logits)  # [batch_size, num_tasks]

                # Aggregate task probabilities into a single confidence score
                # Method 1: Average of maximum probabilities across tasks
                max_probs_per_task = torch.maximum(sigmoid_probs, 1 - sigmoid_probs)  # Max of prob and 1-prob
                avg_max_prob = max_probs_per_task.mean(dim=1)  # Average across tasks

                # Convert to OOD scores (lower confidence = higher OOD score)
                ood_scores = 1.0 - avg_max_prob
            else:
                # Standard single-task or binary classification
                if logits.shape[1] == 1:
                    # Single output (binary classification with BCE)
                    sigmoid_probs = torch.sigmoid(logits)
                    max_probs = torch.maximum(sigmoid_probs, 1 - sigmoid_probs).squeeze()
                else:
                    # Multi-class with softmax
                    probs = F.softmax(logits, dim=1)
                    max_probs = probs.max(dim=1)[0]

                ood_scores = 1.0 - max_probs

        return ood_scores
    
    def _odin_score(self, features, device, temperature=10.0, noise_magnitude=0.0014):
        """ODIN: Out-of-DIstribution detector for Neural networks."""
        # Get hyperparameters from args if available
        if hasattr(self.args, 'T'):
            temperature = self.args.T
        if hasattr(self.args, 'noise'):
            noise_magnitude = self.args.noise

        # Use eval mode to freeze BatchNorm statistics (recommended for ODIN)
        # This prevents BatchNorm from normalizing gradients during backward pass
        original_mode = self.model.training
        self.model.eval()  # Changed from self.model.train() to self.model.eval()

        try:
            # Enable gradients for input perturbation
            features = features.clone().detach().requires_grad_(True)

            # Ensure features have the correct dtype
            features = features.float()

            # Forward pass
            logits = self.model(features)

            # Temperature scaling
            logits = logits / temperature

            # Get predicted labels
            pred_labels = logits.argmax(dim=1)

            # Compute loss w.r.t. predicted labels
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, pred_labels)

            # Clear any existing gradients
            if features.grad is not None:
                features.grad.zero_()

            # Compute gradients w.r.t. input
            loss.backward(retain_graph=False)

            # Get gradients - check if None before using
            gradients = features.grad
            if gradients is None:
                raise RuntimeError("ODIN method failed: gradients are None.")

            # Standard ODIN perturbation: x_adv = x - eps * sign(grad_x CE)
            gradient_sign = torch.sign(gradients)
            perturbation = -noise_magnitude * gradient_sign

            # Add perturbations
            perturbed_features = features.detach() + perturbation

            # Second forward pass with perturbed input (no gradients needed)
            with torch.no_grad():
                perturbed_logits = self.model(perturbed_features)
                # Apply temperature scaling exactly once
                perturbed_logits = perturbed_logits / temperature

                # Compute OOD scores - adapted for multi-task
                if perturbed_logits.shape[1] > 2:  # Multi-task case
                    sigmoid_probs = torch.sigmoid(perturbed_logits)
                    max_probs_per_task = torch.maximum(sigmoid_probs, 1 - sigmoid_probs)
                    max_probs = max_probs_per_task.mean(dim=1)
                else:
                    # Single-task case
                    if perturbed_logits.shape[1] == 1:
                        sigmoid_probs = torch.sigmoid(perturbed_logits)
                        max_probs = torch.maximum(sigmoid_probs, 1 - sigmoid_probs).squeeze()
                    else:
                        probs = F.softmax(perturbed_logits, dim=1)
                        max_probs = probs.max(dim=1)[0]

                ood_scores = 1.0 - max_probs

            return ood_scores

        finally:
            # Restore original model mode
            self.model.train(original_mode)
    
    def _energy_score(self, features):
        """Energy-based OOD score - adapted for multi-task."""
        with torch.no_grad():
            logits = self.model(features)

            # Check if this is multi-task
            if logits.shape[1] > 2:  # Multi-task case
                # For multi-task with sigmoid outputs, compute energy differently
                # Energy = -log(sum(exp(logits))) for each task, then aggregate

                # Method 1: Average energy across tasks
                task_energies = []
                for task_idx in range(logits.shape[1]):
                    task_logits = logits[:, task_idx:task_idx+1]  # Keep dimension
                    # For binary task, create complementary logits
                    complementary_logits = torch.cat([task_logits, -task_logits], dim=1)
                    task_energy = -torch.logsumexp(complementary_logits, dim=1)
                    task_energies.append(task_energy)

                # Average energy across all tasks
                energy = torch.stack(task_energies, dim=1).mean(dim=1)
            else:
                # Standard energy computation for single-task or binary classification
                if logits.shape[1] == 1:
                    # Single output - create complementary logits for energy computation
                    complementary_logits = torch.cat([logits, -logits], dim=1)
                    energy = -torch.logsumexp(complementary_logits, dim=1)
                else:
                    # Multi-class case
                    energy = -torch.logsumexp(logits, dim=1)

            # Higher energy = more likely OOD
            ood_scores = energy

        return ood_scores
    
    def _mahalanobis_score(self, features, device):
        """Mahalanobis distance-based OOD detection - adapted for multi-task."""
        if self.class_means is None or self.precision_matrix is None:
            raise ValueError("Mahalanobis method requires pre-computed class statistics. "
                           "Call fit_class_statistics_from_dataloader() first.")

        with torch.no_grad():
            # Get feature representations (use penultimate layer features)
            if hasattr(self.model, 'get_features'):
                hidden_features = self.model.get_features(features)
            else:
                # Fallback: use features directly
                hidden_features = features

            batch_size = hidden_features.size(0)

            # For multi-task, we compute distance to overall data distribution
            # rather than class-specific distributions
            if hasattr(self, 'is_multitask') and getattr(self, 'is_multitask', False):
                # Single global mean and covariance for multi-task
                if len(self.class_means) == 1:
                    # Use global statistics
                    global_mean = self.class_means[0]
                    centered = hidden_features - global_mean.unsqueeze(0)
                else:
                    # Average of class means as global mean
                    global_mean = torch.stack(list(self.class_means.values())).mean(dim=0)
                    centered = hidden_features - global_mean.unsqueeze(0)

                # Mahalanobis distance: sqrt((x-μ)^T Σ^(-1) (x-μ))
                mahal_dist = torch.sqrt(
                    torch.sum(centered @ self.precision_matrix * centered, dim=1)
                )
                ood_scores = mahal_dist
            else:
                # Standard single-task: minimum distance to any class
                min_distances = torch.full((batch_size,), float('inf'), device=device)

                for class_idx in range(self.num_classes):
                    class_mean = self.class_means[class_idx]

                    # Center features
                    centered = hidden_features - class_mean.unsqueeze(0)
                
                    # Compute Mahalanobis distance: (x-μ)ᵀ Σ⁻¹ (x-μ)
                    distances = torch.sum((centered @ self.precision_matrix) * centered, dim=1)

                    # Keep minimum distance
                    min_distances = torch.min(min_distances, distances)

                ood_scores = min_distances

            return ood_scores
    
    def fit_class_statistics_from_dataloader(self, data_loader, device, num_classes):
        """
        Compute class/global statistics for Mahalanobis method from a DataLoader.

        - Single-task classification: per-class means + shared precision.
        - Multi-task (e.g., GOOD-PCBA with 128 tasks): one global mean + shared precision.

        Args:
            data_loader: DataLoader containing training data with features and labels
            device: torch device
            num_classes: Number of output dimensions in the trained head
        """
        logger.info("Computing class statistics for Mahalanobis method...")

        self.num_classes = num_classes

        # Collect features and labels
        all_features = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting features"):
                features = batch['features'].to(device)
                labels = batch['labels']  # may be multi-dim; keep on CPU for shape inspection

                # Get hidden representations
                hidden_features = self.model.get_features(features)

                all_features.append(hidden_features.cpu())
                all_labels.append(labels.cpu())

        all_features = torch.cat(all_features, dim=0)  # [N, D']
        all_labels = torch.cat(all_labels, dim=0)      # [N] or [N, T]

        is_multitask = all_labels.dim() > 1 and all_labels.size(-1) > 1
        logger.info(
            f"Collected {len(all_features)} feature vectors; features shape={all_features.shape}, labels shape={all_labels.shape}; multitask={is_multitask}"
        )

        if is_multitask:
            # Multi-task: compute a single global mean and shared precision over all features
            # Avoid masking/boolean indexing mismatch by not trying to build per-class subsets
            try:
                global_mean = all_features.mean(dim=0)  # [D']
                centered = (all_features - global_mean).numpy()

                # Robust covariance with shrinkage for stability
                try:
                    lw = sklearn.covariance.LedoitWolf(assume_centered=True)
                    lw.fit(centered)
                    precision_np = lw.precision_
                    method_used = 'LedoitWolf'
                except Exception:
                    emp_cov = sklearn.covariance.EmpiricalCovariance(assume_centered=True)
                    emp_cov.fit(centered)
                    precision_np = emp_cov.precision_
                    method_used = 'EmpiricalCovariance'

                precision = torch.from_numpy(precision_np).float().to(device)
                feat_dim = precision.size(0)
                precision += 1e-6 * torch.eye(feat_dim, device=device)

                # Store
                self.class_means = torch.stack([global_mean.to(device)])  # list with one mean
                self.precision_matrix = precision
                self.is_multitask = True

                logger.info(
                    f"Computed global mean/precision for multi-task (tasks={all_labels.size(-1)}), method={method_used}"
                )
            except Exception as e:
                logger.warning(f"Failed global covariance estimation for multi-task: {e}; using identity precision")
                feat_dim = all_features.size(1)
                self.class_means = torch.stack([all_features.mean(dim=0).to(device)])
                self.precision_matrix = torch.eye(feat_dim, device=device)
                self.is_multitask = True

            logger.info("Mahalanobis statistics fitted for multi-task setting")
            return

        # Single-task classification (labels should be 1D category indices)
        if all_labels.dim() > 1:
            # Attempt to squeeze any trailing size-1 dimensions
            squeezed = all_labels.squeeze()
            if squeezed.dim() != 1:
                raise ValueError(
                    f"Expected 1D labels for single-task, got shape {all_labels.shape} after squeeze -> {squeezed.shape}"
                )
            all_labels = squeezed

        logger.info(f"Proceeding with per-class statistics for {num_classes} classes")

        # Compute class means
        class_means = []
        class_features_list = []

        for class_idx in range(num_classes):
            class_mask = (all_labels == class_idx)
            class_count = int(class_mask.sum().item())

            if class_count > 0:
                class_features = all_features[class_mask]
                class_mean = class_features.mean(dim=0)
                class_means.append(class_mean)
                class_features_list.append(class_features)
                logger.info(f"Class {class_idx}: {class_count} samples")
            else:
                logger.warning(f"No samples found for class {class_idx}, using zero mean")
                class_means.append(torch.zeros(all_features.size(1)))
                class_features_list.append(torch.zeros(1, all_features.size(1)))

        self.class_means = torch.stack(class_means).to(device)

        # Compute precision matrix (inverse covariance) from centered features across classes
        centered_features = []
        for class_idx in range(num_classes):
            if len(class_features_list[class_idx]) > 1:
                centered = class_features_list[class_idx] - class_means[class_idx]
                centered_features.append(centered)

        if centered_features:
            all_centered = torch.cat(centered_features, dim=0).numpy()

            try:
                # Prefer Ledoit-Wolf shrinkage for stability in high dimensions
                try:
                    lw = sklearn.covariance.LedoitWolf(assume_centered=True)
                    lw.fit(all_centered)
                    precision_np = lw.precision_
                    method_used = 'LedoitWolf'
                except Exception:
                    emp_cov = sklearn.covariance.EmpiricalCovariance(assume_centered=True)
                    emp_cov.fit(all_centered)
                    precision_np = emp_cov.precision_
                    method_used = 'EmpiricalCovariance'

                precision = torch.from_numpy(precision_np).float().to(device)

                # Add small regularization for numerical stability
                feat_dim = precision.size(0)
                precision += 1e-6 * torch.eye(feat_dim, device=device)

                self.precision_matrix = precision
                logger.info(f"Successfully computed precision matrix for Mahalanobis method using {method_used}")

            except Exception as e:
                logger.warning(f"Failed to compute precision matrix: {e}, using identity")
                feat_dim = all_features.size(1)
                self.precision_matrix = torch.eye(feat_dim, device=device)
        else:
            logger.warning("Insufficient data for covariance estimation, using identity matrix")
            feat_dim = all_features.size(1)
            self.precision_matrix = torch.eye(feat_dim, device=device)

        logger.info("Class statistics fitting completed")
    
    def fit_class_statistics_direct(self, features, labels, num_classes, device):
        """
        Alternative method to fit class statistics directly from feature tensors.
        
        Args:
            features: Feature tensor [N, feature_dim]
            labels: Label tensor [N]
            num_classes: Number of classes
            device: torch device
        """
        logger.info("Computing class statistics directly from feature tensors...")
        
        self.num_classes = num_classes
        features = features.to(device)
        labels = labels.to(device)
        
        # Compute class means
        class_means = []
        class_features_list = []
        
        for class_idx in range(num_classes):
            class_mask = (labels == class_idx)
            class_count = class_mask.sum().item()
            
            if class_count > 0:
                class_features = features[class_mask]
                class_mean = class_features.mean(dim=0)
                class_means.append(class_mean)
                class_features_list.append(class_features)
                logger.info(f"Class {class_idx}: {class_count} samples")
            else:
                logger.warning(f"No samples for class {class_idx}")
                class_means.append(torch.zeros(features.size(1), device=device))
                class_features_list.append(torch.zeros(1, features.size(1), device=device))
        
        self.class_means = torch.stack(class_means)
        
        # Compute precision matrix
        centered_features = []
        for class_idx in range(num_classes):
            if len(class_features_list[class_idx]) > 1:
                centered = class_features_list[class_idx] - class_means[class_idx]
                centered_features.append(centered)
        
        if centered_features:
            all_centered = torch.cat(centered_features, dim=0).cpu().numpy()
            
            try:
                # Prefer Ledoit-Wolf shrinkage; fallback to empirical covariance
                try:
                    lw = sklearn.covariance.LedoitWolf(assume_centered=True)
                    lw.fit(all_centered)
                    precision_np = lw.precision_
                    method_used = 'LedoitWolf'
                except Exception:
                    emp_cov = sklearn.covariance.EmpiricalCovariance(assume_centered=True)
                    emp_cov.fit(all_centered)
                    precision_np = emp_cov.precision_
                    method_used = 'EmpiricalCovariance'

                precision = torch.from_numpy(precision_np).float().to(device)
                
                # Regularization
                feat_dim = precision.size(0)
                precision += 1e-6 * torch.eye(feat_dim, device=device)
                
                self.precision_matrix = precision
                logger.info(f"Direct class statistics fitting completed successfully using {method_used}")
                
            except Exception as e:
                logger.warning(f"Direct precision computation failed: {e}")
                feat_dim = features.size(1)
                self.precision_matrix = torch.eye(feat_dim, device=device)
        else:
            feat_dim = features.size(1)
            self.precision_matrix = torch.eye(feat_dim, device=device)

    def _knn_score(self, features, k=50):
        """KNN-based OOD detection using distance to k nearest neighbors."""
        if self.train_features is None:
            raise ValueError("KNN method requires training features. "
                           "Call store_training_features() first.")

        with torch.no_grad():
            # Get feature representations
            if hasattr(self.model, 'get_features'):
                hidden_features = self.model.get_features(features)
            else:
                # Fallback: use features directly
                hidden_features = features

            # Convert to numpy and apply stored normalization
            test_features_np = hidden_features.cpu().numpy()
            if self._feat_mean is not None and self._feat_std is not None:
                test_features_np = (test_features_np - self._feat_mean) / self._feat_std

            # Build KNN model if not already built
            if self.knn_model is None or self.knn_model.n_neighbors != k:
                logger.info(f"Building KNN model with k={k} using {len(self.train_features)} training samples")
                self.knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='auto')
                self.knn_model.fit(self.train_features)

            # Compute distances to k nearest neighbors
            distances, indices = self.knn_model.kneighbors(test_features_np)

            # Use mean distance to k nearest neighbors as OOD score
            knn_scores = distances.mean(axis=1)

            return torch.from_numpy(knn_scores).float()

    def _lof_score(self, features, n_neighbors=20):
        """LOF-based OOD detection using Local Outlier Factor (novelty mode).

        Fits once on training features (no test leakage) and scores test batches.
        """
        if self.train_features is None:
            raise ValueError("LOF method requires training features. "
                           "Call store_training_features() first.")

        with torch.no_grad():
            # Get feature representations
            if hasattr(self.model, 'get_features'):
                hidden_features = self.model.get_features(features)
            else:
                # Fallback: use features directly
                hidden_features = features

            # Convert to numpy and apply stored normalization
            test_features_np = hidden_features.cpu().numpy()
            if self._feat_mean is not None and self._feat_std is not None:
                test_features_np = (test_features_np - self._feat_mean) / self._feat_std

            # Build and fit LOF model once on training features (novelty detection)
            if (
                self.lof_model is None or
                getattr(self.lof_model, 'n_neighbors', None) != n_neighbors
            ):
                # Cap n_neighbors to be < number of training samples
                nn_used = min(n_neighbors, max(1, len(self.train_features) - 1))
                logger.info(
                    f"Building LOF model (novelty=True) with n_neighbors={nn_used} "
                    f"using {len(self.train_features)} training samples"
                )
                self.lof_model = LocalOutlierFactor(
                    n_neighbors=nn_used,
                    contamination='auto',
                    novelty=True  # Fit on train once; score novel points
                )
                self.lof_model.fit(self.train_features)

            # score_samples: lower = more abnormal. Convert so higher = more OOD
            scores = -self.lof_model.score_samples(test_features_np)
            return torch.from_numpy(scores).float()

    def store_training_features(self, dataloader, device):
        """Store training features for KNN and LOF methods."""
        logger.info("Storing training features for KNN and LOF methods...")

        all_features = []
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting training features"):
                if isinstance(batch, dict):
                    features = batch['features'].to(device)
                else:
                    features = batch[0].to(device)

                # Get feature representations
                if hasattr(self.model, 'get_features'):
                    hidden_features = self.model.get_features(features)
                else:
                    # Fallback: use features directly
                    hidden_features = features

                all_features.append(hidden_features.cpu().numpy())

        # Concatenate all features
        self.train_features = np.vstack(all_features)

        # Compute and apply standardization to stabilize distance/density metrics
        eps = 1e-6
        self._feat_mean = self.train_features.mean(axis=0)
        self._feat_std = self.train_features.std(axis=0)
        # Avoid division by zero
        self._feat_std = np.where(self._feat_std < eps, 1.0, self._feat_std)
        self.train_features = (self.train_features - self._feat_mean) / self._feat_std

        logger.info(
            f"Stored {len(self.train_features)} training feature vectors of dimension {self.train_features.shape[1]} "
            f"(features standardized)"
        )

    def _mc_dropout_score(self, features, n_samples=20, metric='entropy'):
        """Monte Carlo Dropout-based OOD detection.

        Uses dropout at test time to estimate predictive uncertainty.
        Higher uncertainty indicates more likely OOD.

        Args:
            features: Input feature tensor [batch_size, input_dim]
            n_samples: Number of forward passes with dropout (default: 20)
            metric: Uncertainty metric - 'entropy' or 'variance' (default: 'entropy')

        Returns:
            scores: OOD scores [batch_size] (higher = more uncertain = more OOD)
        """
        # Store original model state
        original_mode = self.model.training

        # IMPORTANT: Keep model in eval mode (so BatchNorm uses fixed stats)
        # but enable ONLY Dropout layers for MC sampling
        self.model.eval()

        # Enable dropout layers while keeping other layers (BatchNorm, etc.) in eval mode
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()

        self.model.apply(enable_dropout)

        predictions = []
        output_dim = None  # Will store the output dimension for task type detection

        # Multiple forward passes with dropout enabled
        with torch.no_grad():
            for i in range(n_samples):
                logits = self.model(features)

                # Save output dimension on first iteration
                if i == 0:
                    output_dim = logits.shape[1]

                # Convert logits to probabilities based on task type
                if output_dim > 2:  # Multi-task case (Sigmoid)
                    probs = torch.sigmoid(logits)
                elif output_dim == 1:  # Binary with single output
                    probs = torch.sigmoid(logits)
                    probs = torch.cat([1 - probs, probs], dim=1)  # [batch_size, 2]
                else:  # Multi-class with softmax (including binary classification)
                    probs = F.softmax(logits, dim=1)

                predictions.append(probs)

        # Restore original model state
        self.model.train(original_mode)

        # Stack predictions: [n_samples, batch_size, num_classes]
        predictions = torch.stack(predictions)

        # Compute uncertainty based on metric
        if metric == 'entropy':
            # Predictive entropy: entropy of the mean prediction
            mean_probs = predictions.mean(dim=0)  # [batch_size, num_classes/num_tasks]
            epsilon = 1e-10  # For numerical stability

            # Compute entropy based on task type
            if output_dim > 2:  # Multi-task with Sigmoid
                # Binary entropy for each task: H(p) = -[p*log(p) + (1-p)*log(1-p)]
                binary_entropy = -(mean_probs * torch.log(mean_probs + epsilon) +
                                  (1 - mean_probs) * torch.log(1 - mean_probs + epsilon))
                # Average entropy across all tasks
                entropy = binary_entropy.mean(dim=1)
            else:  # Single-task or binary classification with Softmax
                # Standard entropy: H(p) = -Σ p*log(p)
                entropy = -(mean_probs * torch.log(mean_probs + epsilon)).sum(dim=1)

            ood_scores = entropy

        elif metric == 'variance':
            # Variance of predictions across samples
            # Use variance of max probabilities
            max_probs = predictions.max(dim=2)[0]  # [n_samples, batch_size]
            variance = max_probs.var(dim=0)  # [batch_size]

            ood_scores = variance

        else:
            raise ValueError(f"Unknown MC Dropout metric: {metric}")

        return ood_scores

    def fit_conformal_calibration(self, dataloader, device):
        """Compute calibration scores for Conformal Prediction method.

        Uses validation/calibration data (ID only) to establish the distribution
        of conformity scores (max_prob) under the in-distribution.

        Args:
            dataloader: Calibration data loader (ID samples only)
            device: torch device
        """
        logger.info("Computing conformal calibration scores...")

        all_scores = []
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calibration"):
                if isinstance(batch, dict):
                    features = batch['features'].to(device)
                else:
                    features = batch[0].to(device)

                logits = self.model(features)

                # Compute conformity scores: max_prob (higher = more confident = more conformal)
                if logits.shape[1] > 2:  # Multi-task case
                    # Use sigmoid for multi-task
                    sigmoid_probs = torch.sigmoid(logits)
                    # For each task, get max of (prob, 1-prob), then average
                    max_probs_per_task = torch.maximum(sigmoid_probs, 1 - sigmoid_probs)
                    max_probs = max_probs_per_task.mean(dim=1)
                elif logits.shape[1] == 1:
                    # Binary with single output
                    sigmoid_probs = torch.sigmoid(logits).squeeze()
                    max_probs = torch.maximum(sigmoid_probs, 1 - sigmoid_probs)
                else:
                    # Multi-class with softmax
                    probs = F.softmax(logits, dim=1)
                    max_probs = probs.max(dim=1)[0]

                # Store conformity scores directly (higher confidence = more conformal)
                conformity_scores = max_probs
                all_scores.append(conformity_scores.cpu())

        # Concatenate all calibration scores
        self.calibration_scores = torch.cat(all_scores)

        logger.info(
            f"Calibration complete: {len(self.calibration_scores)} samples, "
            f"mean conformity = {self.calibration_scores.mean():.4f}"
        )

    def _conformal_score(self, features):
        """Conformal Prediction-based OOD detection.

        Computes p-values for test samples based on calibration distribution.
        Lower p-value (less conformal) indicates more likely OOD.

        Args:
            features: Input feature tensor [batch_size, input_dim]

        Returns:
            scores: OOD scores [batch_size] (higher = less conformal = more OOD)
        """
        if self.calibration_scores is None:
            raise ValueError(
                "Conformal method requires calibration scores. "
                "Call fit_conformal_calibration() first."
            )

        with torch.no_grad():
            logits = self.model(features)

            # Compute conformity scores for test samples (same as calibration)
            if logits.shape[1] > 2:  # Multi-task case
                sigmoid_probs = torch.sigmoid(logits)
                max_probs_per_task = torch.maximum(sigmoid_probs, 1 - sigmoid_probs)
                max_probs = max_probs_per_task.mean(dim=1)
            elif logits.shape[1] == 1:
                # Binary with single output
                sigmoid_probs = torch.sigmoid(logits).squeeze()
                max_probs = torch.maximum(sigmoid_probs, 1 - sigmoid_probs)
            else:
                # Multi-class with softmax
                probs = F.softmax(logits, dim=1)
                max_probs = probs.max(dim=1)[0]

            # Conformity scores for test samples (higher = more confident)
            test_conformity = max_probs

            # Compute p-values
            batch_size = features.size(0)
            p_values = torch.zeros(batch_size)

            # Move calibration scores to same device as test scores
            calib_scores = self.calibration_scores.to(test_conformity.device)
            n_calib = len(calib_scores)

            for i in range(batch_size):
                # Count how many calibration scores are <= test score
                # (lower calibration conformity = test sample is more conformal than calibration)
                n_less_equal = (calib_scores <= test_conformity[i]).sum().item()

                # Compute conformal p-value
                # High p-value means test sample is as conformal as calibration (likely ID)
                # Low p-value means test sample is less conformal (likely OOD)
                p_values[i] = (1 + n_less_equal) / (1 + n_calib)

            # OOD score: 1 - p_value
            # High p-value (very conformal/confident) → low OOD score (likely ID)
            # Low p-value (not conformal/uncertain) → high OOD score (likely OOD)
            ood_scores = 1.0 - p_values

        return ood_scores
