"""
Knowledge distillation implementation for model optimization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertConfig, DistilBertForSequenceClassification
from tqdm import tqdm


def create_student_model(teacher_model, num_layers=2, num_labels=2):
    """
    Create a smaller student model based on the teacher architecture
    with fewer transformer layers
    
    Args:
        teacher_model: The teacher model to distill from
        num_layers (int): Number of transformer layers for the student (default: 2)
        num_labels (int): Number of classification labels
        
    Returns:
        student_model: A smaller version of the teacher model
    """
    # Get the config from the teacher and modify it
    if hasattr(teacher_model, 'config'):
        config = DistilBertConfig.from_pretrained(teacher_model.config._name_or_path)
    else:
        # Default config if teacher doesn't have one
        config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
    
    # Modify the config to make a smaller model
    config.n_layers = num_layers  # Reduce the number of layers
    config.num_labels = num_labels
    
    # Create student model from modified config
    student_model = DistilBertForSequenceClassification(config)
    
    return student_model


class DistillationTrainer:
    def __init__(self, teacher_model, student_model, temperature=2.0):
        """
        Trainer for knowledge distillation
        
        Args:
            teacher_model: The larger, well-trained model
            student_model: The smaller model to train
            temperature (float): Temperature parameter for softening probability distributions
        """
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
    
    def distillation_loss(self, student_logits, teacher_logits, labels, alpha=0.5):
        """
        Compute the distillation loss: a weighted average of the hard cross-entropy loss
        and the soft distillation loss (KL-divergence between soft student and teacher outputs)
        
        Args:
            student_logits: Logits from the student model
            teacher_logits: Logits from the teacher model
            labels: True labels
            alpha (float): Weight for hard loss vs soft loss
            
        Returns:
            loss: The combined distillation loss
        """
        # Hard loss: standard cross-entropy with true labels
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Soft loss: KL divergence between soft student and teacher outputs
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        # Combined loss
        return alpha * hard_loss + (1 - alpha) * soft_loss
    
    def train(self, train_dataloader, eval_dataloader, optimizer, device, 
              epochs=3, alpha=0.5, eval_every=100, prepare_batch_fn=None):
        """
        Train the student model using knowledge distillation
        
        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation
            optimizer: Optimizer for training
            device: Device to train on (cuda/cpu)
            epochs (int): Number of training epochs
            alpha (float): Weight for hard loss vs soft loss in distillation
            eval_every (int): Evaluate and print metrics every N steps
            prepare_batch_fn: Function to prepare batch for model input
            
        Returns:
            dict: Training statistics
        """
        self.teacher.to(device)
        self.student.to(device)
        
        # Set models to appropriate modes
        self.teacher.eval()  # Teacher is frozen
        self.student.train()
        
        # Track metrics
        stats = {
            'train_losses': [],
            'eval_accuracies': [],
        }
        
        total_steps = epochs * len(train_dataloader)
        global_step = 0
        
        # Training loop
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
                # Prepare the batch
                if prepare_batch_fn:
                    inputs = prepare_batch_fn(batch, device)
                else:
                    inputs = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass with the student model
                student_outputs = self.student(**inputs)
                student_logits = student_outputs.logits
                
                # Get the teacher's predictions (no grad needed)
                with torch.no_grad():
                    teacher_outputs = self.teacher(**inputs)
                    teacher_logits = teacher_outputs.logits
                
                # Compute the distillation loss
                loss = self.distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=inputs['labels'],
                    alpha=alpha
                )
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track loss
                epoch_losses.append(loss.item())
                stats['train_losses'].append(loss.item())
                
                global_step += 1
                
                # Evaluation during training
                if global_step % eval_every == 0 or global_step == total_steps:
                    accuracy = self.evaluate(eval_dataloader, device, prepare_batch_fn)
                    stats['eval_accuracies'].append(accuracy)
                    
                    print(f"Step {global_step}/{total_steps}, Loss: {loss.item():.4f}, Eval Accuracy: {accuracy:.4f}")
                    
                    # Set student back to training mode
                    self.student.train()
            
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")
        
        return stats
    
    def evaluate(self, eval_dataloader, device, prepare_batch_fn=None):
        """
        Evaluate the student model
        
        Args:
            eval_dataloader: DataLoader for evaluation data
            device: Device to run evaluation on
            prepare_batch_fn: Function to prepare batch for model input
            
        Returns:
            float: Accuracy of the student model
        """
        self.student.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Prepare the batch
                if prepare_batch_fn:
                    inputs = prepare_batch_fn(batch, device)
                else:
                    inputs = {k: v.to(device) for k, v in batch.items()}
                
                # Get student predictions
                outputs = self.student(**inputs)
                predictions = outputs.logits.argmax(dim=-1)
                
                # Compute accuracy
                correct += (predictions == inputs['labels']).sum().item()
                total += inputs['labels'].size(0)
        
        return correct / total if total > 0 else 0
