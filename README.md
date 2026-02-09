# Deep Learning Repository

This repository contains a comprehensive collection of Jupyter notebooks covering fundamental concepts and advanced techniques in Deep Learning.

## Repository Capabilities

This repository provides educational materials and implementations for:

### Neural Network Fundamentals
- **Backpropagation**: Understanding and implementing the backpropagation algorithm
- **Weight Initialization**: Techniques for properly initializing neural network weights
- **Gradient Computation**: Chain rule and automatic differentiation

### Optimization Techniques
- **Line Search Methods**: Finding optimal step sizes in optimization
- **Gradient Descent**: Basic gradient-based optimization
- **Stochastic Gradient Descent (SGD)**: Mini-batch optimization techniques
- **Momentum**: Accelerated gradient descent with momentum
- **Adam Optimizer**: Adaptive moment estimation optimizer

### Advanced Topics
- **Bias-Variance Trade-off**: Understanding model complexity and generalization
- **Double Descent**: Modern phenomena in deep learning generalization
- **High-Dimensional Spaces**: Challenges and properties of high-dimensional data
- **MNIST 1D Performance**: Evaluating neural networks on simplified datasets

## Repository Structure

### `/neural network optimization/`
Contains notebooks focused on optimization algorithms:
- `6_1_Line_Search.ipynb` - Line search methods for optimization
- `6_2_Gradient_Descent.ipynb` - Basic gradient descent implementation
- `6_3_Stochastic_Gradient_Descent.ipynb` - SGD with mini-batches
- `6_4_Momentum.ipynb` - Momentum-based optimization
- `6_5_Adam.ipynb` - Adam optimizer implementation

### `/lab-7/`
Contains advanced notebooks on backpropagation and training:
- `7_1_Backpropagation_in_Toy_Model.ipynb` - Simple backprop example
- `7_2_Backpropagation.ipynb` - Full backpropagation implementation
- `7_3_Initialization.ipynb` - Weight initialization strategies
- `8_1_MNIST_1D_Performance.ipynb` - Model evaluation on MNIST 1D
- `8_2_Bias_Variance_Trade_Off.ipynb` - Model complexity analysis
- `8_3_Double_Descent.ipynb` - Modern generalization phenomena
- `8_4_High_Dimensional_Spaces.ipynb` - High-dimensional data properties

### `/lab-7/backprop-hyperopt/`
Additional materials on backpropagation:
- `backprop_chain_rule_tutorial.ipynb` - Chain rule tutorial
- `micrograd_lecture_first_half_roughly.ipynb` - Micrograd-style implementation
- `output.svg` - Visualization output

## Features and Capabilities

### Educational Content
- **Step-by-step explanations** of deep learning concepts
- **Visual demonstrations** of algorithms in action
- **Interactive Jupyter notebooks** for hands-on learning

### Implementations
- **From-scratch implementations** to understand fundamentals
- **Modern optimization algorithms** (SGD, Momentum, Adam)
- **Visualization tools** for understanding high-dimensional spaces

### Topics Covered
1. **Forward and Backward Propagation**: Complete implementation and visualization
2. **Optimization Algorithms**: Multiple optimization strategies with comparisons
3. **Hyperparameter Tuning**: Understanding how parameters affect training
4. **Generalization**: Bias-variance tradeoff and double descent phenomenon
5. **Initialization Strategies**: Impact of weight initialization on training

## Getting Started

### Prerequisites
- Python 3.x
- Jupyter Notebook or JupyterLab
- Common deep learning libraries (NumPy, Matplotlib, etc.)

### Installation
```bash
# Clone the repository (replace <username> with the repository owner)
git clone https://github.com/<username>/DL.git
cd DL

# Install required packages (if requirements.txt exists)
pip install jupyter numpy matplotlib

# Launch Jupyter
jupyter notebook
```

### Usage
1. Navigate to the desired topic folder
2. Open the notebook in Jupyter
3. Run cells sequentially to see explanations and implementations
4. Experiment with parameters to deepen understanding

## Learning Path

Recommended order for beginners:
1. Start with **Gradient Descent** (`6_2_Gradient_Descent.ipynb`)
2. Progress to **Stochastic Gradient Descent** (`6_3_Stochastic_Gradient_Descent.ipynb`)
3. Learn **Backpropagation in Toy Model** (`7_1_Backpropagation_in_Toy_Model.ipynb`)
4. Understand **Full Backpropagation** (`7_2_Backpropagation.ipynb`)
5. Explore **Advanced Optimizers** (Momentum, Adam)
6. Study **Generalization** (Bias-Variance, Double Descent)

## Contributing

This repository serves as an educational resource. Feel free to:
- Report issues or errors in notebooks
- Suggest additional topics or improvements
- Share your own implementations or variations

## License

This repository is provided for educational purposes. Please contact the repository owner regarding usage terms and licensing.

## Acknowledgments

This repository contains educational materials for learning deep learning fundamentals and advanced techniques.
