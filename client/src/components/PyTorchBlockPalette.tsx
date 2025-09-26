import React from 'react';
import { PyTorchBlock } from './PyTorchBlock';

const PYTORCH_BLOCKS = [
  // Input/Output Layers
  {
    id: 'input',
    label: 'Input Layer',
    category: 'Input/Output',
    color: '#4CAF50',
    icon: '📥',
    config: { shape: [1, 784] }
  },
  {
    id: 'output',
    label: 'Output Layer', 
    category: 'Input/Output',
    color: '#4CAF50',
    icon: '📤',
    config: { classes: 10 }
  },
  
  // Linear Layers
  {
    id: 'linear',
    label: 'Linear/Dense',
    category: 'Linear Layers',
    color: '#2196F3',
    icon: '🔗',
    config: { input_features: 784, output_features: 128 }
  },
  {
    id: 'bilinear',
    label: 'Bilinear',
    category: 'Linear Layers', 
    color: '#2196F3',
    icon: '🔗',
    config: { in1_features: 128, in2_features: 128, out_features: 64 }
  },

  // Convolution Layers
  {
    id: 'conv1d',
    label: 'Conv1d',
    category: 'Convolution',
    color: '#FF9800',
    icon: '📊',
    config: { in_channels: 1, out_channels: 32, kernel_size: 3 }
  },
  {
    id: 'conv2d',
    label: 'Conv2d',
    category: 'Convolution',
    color: '#FF9800',
    icon: '🖼️',
    config: { in_channels: 3, out_channels: 64, kernel_size: 3 }
  },
  {
    id: 'conv3d',
    label: 'Conv3d',
    category: 'Convolution',
    color: '#FF9800', 
    icon: '🎯',
    config: { in_channels: 3, out_channels: 64, kernel_size: 3 }
  },

  // Pooling Layers
  {
    id: 'maxpool1d',
    label: 'MaxPool1d',
    category: 'Pooling',
    color: '#9C27B0',
    icon: '⬇️',
    config: { kernel_size: 2 }
  },
  {
    id: 'maxpool2d',
    label: 'MaxPool2d',
    category: 'Pooling',
    color: '#9C27B0',
    icon: '⬇️',
    config: { kernel_size: 2, stride: 2 }
  },
  {
    id: 'avgpool2d',
    label: 'AvgPool2d',
    category: 'Pooling',
    color: '#9C27B0',
    icon: '📉',
    config: { kernel_size: 2 }
  },

  // Activation Functions
  {
    id: 'relu',
    label: 'ReLU',
    category: 'Activation',
    color: '#F44336',
    icon: '⚡',
    config: {}
  },
  {
    id: 'sigmoid',
    label: 'Sigmoid',
    category: 'Activation',
    color: '#F44336',
    icon: '〰️',
    config: {}
  },
  {
    id: 'tanh',
    label: 'Tanh',
    category: 'Activation',
    color: '#F44336',
    icon: '📈',
    config: {}
  },
  {
    id: 'softmax',
    label: 'Softmax',
    category: 'Activation',
    color: '#F44336',
    icon: '🎯',
    config: { dim: 1 }
  },
  {
    id: 'leaky_relu',
    label: 'LeakyReLU',
    category: 'Activation',
    color: '#F44336',
    icon: '⚡',
    config: { negative_slope: 0.01 }
  },

  // Normalization
  {
    id: 'batchnorm1d',
    label: 'BatchNorm1d',
    category: 'Normalization',
    color: '#607D8B',
    icon: '🔄',
    config: { num_features: 128 }
  },
  {
    id: 'batchnorm2d',
    label: 'BatchNorm2d',
    category: 'Normalization',
    color: '#607D8B',
    icon: '🔄',
    config: { num_features: 64 }
  },
  {
    id: 'layernorm',
    label: 'LayerNorm',
    category: 'Normalization',
    color: '#607D8B',
    icon: '🔄',
    config: { normalized_shape: 128 }
  },

  // Regularization
  {
    id: 'dropout',
    label: 'Dropout',
    category: 'Regularization',
    color: '#795548',
    icon: '❌',
    config: { p: 0.5 }
  },
  {
    id: 'dropout2d',
    label: 'Dropout2d',
    category: 'Regularization',
    color: '#795548',
    icon: '❌',
    config: { p: 0.25 }
  },

  // Loss Functions
  {
    id: 'cross_entropy',
    label: 'CrossEntropyLoss',
    category: 'Loss Functions',
    color: '#E91E63',
    icon: '🎯',
    config: {}
  },
  {
    id: 'mse_loss',
    label: 'MSELoss',
    category: 'Loss Functions',
    color: '#E91E63',
    icon: '📏',
    config: {}
  },
  {
    id: 'bce_loss',
    label: 'BCELoss',
    category: 'Loss Functions',
    color: '#E91E63',
    icon: '⚖️',
    config: {}
  },

  // Optimizers
  {
    id: 'adam',
    label: 'Adam Optimizer',
    category: 'Optimizers',
    color: '#3F51B5',
    icon: '🚀',
    config: { lr: 0.001, weight_decay: 0.0001 }
  },
  {
    id: 'sgd',
    label: 'SGD Optimizer',
    category: 'Optimizers',
    color: '#3F51B5',
    icon: '📈',
    config: { lr: 0.01, momentum: 0.9 }
  },

  // Recurrent Layers
  {
    id: 'lstm',
    label: 'LSTM',
    category: 'Recurrent',
    color: '#009688',
    icon: '🔄',
    config: { input_size: 128, hidden_size: 256, num_layers: 1 }
  },
  {
    id: 'gru',
    label: 'GRU',
    category: 'Recurrent',
    color: '#009688',
    icon: '🔄',
    config: { input_size: 128, hidden_size: 256, num_layers: 1 }
  },

  // Attention
  {
    id: 'multihead_attention',
    label: 'MultiheadAttention',
    category: 'Attention',
    color: '#FF5722',
    icon: '🎯',
    config: { embed_dim: 512, num_heads: 8 }
  }
];

const categories = [...new Set(PYTORCH_BLOCKS.map(block => block.category))];

export function PyTorchBlockPalette() {
  return (
    <div className="pytorch-block-palette">
      <div className="palette-header">
        <h3>PyTorch Blocks</h3>
        <p>Drag blocks to the canvas</p>
      </div>
      
      <div className="palette-content">
        {categories.map(category => (
          <div key={category} className="block-category">
            <h4 className="category-header">{category}</h4>
            <div className="category-blocks">
              {PYTORCH_BLOCKS
                .filter(block => block.category === category)
                .map(block => (
                  <PyTorchBlock
                    key={block.id}
                    id={block.id}
                    label={block.label}
                    color={block.color}
                    icon={block.icon}
                    config={block.config}
                  />
                ))
              }
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}