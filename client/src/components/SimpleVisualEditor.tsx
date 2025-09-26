import React, { useState, useRef } from 'react';

interface BlockNode {
  id: string;
  type: string;
  label: string;
  x: number;
  y: number;
  config: any;
  color: string;
  icon: string;
}

interface Connection {
  from: string;
  to: string;
}

const PYTORCH_BLOCKS = [
  // Input/Output Layers
  { id: 'input', label: 'Input Layer', category: 'Input/Output', color: '#4CAF50', icon: '📥', config: { shape: [1, 784] } },
  { id: 'output', label: 'Output Layer', category: 'Input/Output', color: '#4CAF50', icon: '📤', config: { classes: 10 } },
  
  // Linear Layers
  { id: 'linear', label: 'Linear/Dense', category: 'Linear Layers', color: '#2196F3', icon: '🔗', config: { input_features: 784, output_features: 128 } },
  { id: 'bilinear', label: 'Bilinear', category: 'Linear Layers', color: '#2196F3', icon: '🔗', config: { in1_features: 128, in2_features: 128, out_features: 64 } },

  // Convolution Layers
  { id: 'conv1d', label: 'Conv1d', category: 'Convolution', color: '#FF9800', icon: '📊', config: { in_channels: 1, out_channels: 32, kernel_size: 3 } },
  { id: 'conv2d', label: 'Conv2d', category: 'Convolution', color: '#FF9800', icon: '🖼️', config: { in_channels: 3, out_channels: 64, kernel_size: 3 } },

  // Activation Functions
  { id: 'relu', label: 'ReLU', category: 'Activation', color: '#F44336', icon: '⚡', config: {} },
  { id: 'sigmoid', label: 'Sigmoid', category: 'Activation', color: '#F44336', icon: '〰️', config: {} },
  { id: 'softmax', label: 'Softmax', category: 'Activation', color: '#F44336', icon: '🎯', config: { dim: 1 } },

  // Pooling Layers
  { id: 'maxpool2d', label: 'MaxPool2d', category: 'Pooling', color: '#9C27B0', icon: '⬇️', config: { kernel_size: 2, stride: 2 } },
  { id: 'avgpool2d', label: 'AvgPool2d', category: 'Pooling', color: '#9C27B0', icon: '📉', config: { kernel_size: 2 } },

  // Normalization
  { id: 'batchnorm2d', label: 'BatchNorm2d', category: 'Normalization', color: '#607D8B', icon: '🔄', config: { num_features: 64 } },
  
  // Regularization
  { id: 'dropout', label: 'Dropout', category: 'Regularization', color: '#795548', icon: '❌', config: { p: 0.5 } },

  // Loss Functions
  { id: 'cross_entropy', label: 'CrossEntropyLoss', category: 'Loss Functions', color: '#E91E63', icon: '🎯', config: {} },
  { id: 'mse_loss', label: 'MSELoss', category: 'Loss Functions', color: '#E91E63', icon: '📏', config: {} },

  // Optimizers
  { id: 'adam', label: 'Adam Optimizer', category: 'Optimizers', color: '#3F51B5', icon: '🚀', config: { lr: 0.001 } },
  { id: 'sgd', label: 'SGD Optimizer', category: 'Optimizers', color: '#3F51B5', icon: '📈', config: { lr: 0.01 } },
];

const categories = [...new Set(PYTORCH_BLOCKS.map(block => block.category))];

export function SimpleVisualEditor() {
  const [selectedProject, setSelectedProject] = useState<string>('');
  const [isProjectCreated, setIsProjectCreated] = useState(false);
  const [nodes, setNodes] = useState<BlockNode[]>([
    {
      id: 'initial-input',
      type: 'input',
      label: 'Input Layer',
      x: 400,
      y: 100,
      config: { shape: [1, 784] },
      color: '#4CAF50',
      icon: '📥'
    }
  ]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [draggedNode, setDraggedNode] = useState<BlockNode | null>(null);
  const [showConfig, setShowConfig] = useState<string | null>(null);
  
  const canvasRef = useRef<HTMLDivElement>(null);

  const handleCreateProject = () => {
    if (selectedProject.trim()) {
      setIsProjectCreated(true);
    }
  };

  const handleDragStart = (e: React.DragEvent, blockData: any) => {
    e.dataTransfer.setData('application/pytorch-block', JSON.stringify(blockData));
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const blockData = JSON.parse(e.dataTransfer.getData('application/pytorch-block'));
    
    if (!canvasRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const newNode: BlockNode = {
      id: `${blockData.id}-${Date.now()}`,
      type: blockData.id,
      label: blockData.label,
      x: Math.max(0, x - 90), // Center the block
      y: Math.max(0, y - 40),
      config: blockData.config,
      color: blockData.color,
      icon: blockData.icon
    };

    setNodes(prev => [...prev, newNode]);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const generatePythonCode = () => {
    let code = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n";
    code += "class GeneratedModel(nn.Module):\n";
    code += "    def __init__(self):\n";
    code += "        super(GeneratedModel, self).__init__()\n";
    
    // Add layers based on nodes
    nodes.forEach((node, index) => {
      if (node.type !== 'input') {
        const layerName = `layer_${index}`;
        code += `        self.${layerName} = ${getPytorchCode(node)}\n`;
      }
    });
    
    code += "\n    def forward(self, x):\n";
    code += "        # Forward pass implementation\n";
    code += "        return x\n\n";
    code += "# Create model instance\nmodel = GeneratedModel()\nprint(model)";
    
    return code;
  };

  const getPytorchCode = (node: BlockNode) => {
    switch (node.type) {
      case 'linear':
        return `nn.Linear(${node.config?.input_features || 784}, ${node.config?.output_features || 128})`;
      case 'conv2d':
        return `nn.Conv2d(${node.config?.in_channels || 1}, ${node.config?.out_channels || 32}, ${node.config?.kernel_size || 3})`;
      case 'relu':
        return 'nn.ReLU()';
      case 'sigmoid':
        return 'nn.Sigmoid()';
      case 'dropout':
        return `nn.Dropout(${node.config?.p || 0.5})`;
      case 'batchnorm2d':
        return `nn.BatchNorm2d(${node.config?.num_features || 64})`;
      case 'maxpool2d':
        return `nn.MaxPool2d(${node.config?.kernel_size || 2})`;
      case 'cross_entropy':
        return 'nn.CrossEntropyLoss()';
      default:
        return 'nn.Identity()';
    }
  };

  if (!isProjectCreated) {
    return (
      <div className="project-creation-screen">
        <div className="project-creation-container">
          <h1>PyTorch Visual Block Editor</h1>
          <p>Create visual PyTorch models with drag-and-drop blocks</p>
          
          <div className="project-form">
            <input
              type="text"
              placeholder="Enter project name..."
              value={selectedProject}
              onChange={(e) => setSelectedProject(e.target.value)}
              className="project-input"
            />
            <button 
              onClick={handleCreateProject}
              className="create-project-btn"
              disabled={!selectedProject.trim()}
            >
              Create Project
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="visual-block-editor">
      <div className="editor-header">
        <h2>Project: {selectedProject}</h2>
        <div className="header-actions">
          <button className="export-btn" onClick={() => {
            const code = generatePythonCode();
            console.log('Generated PyTorch Code:', code);
            navigator.clipboard.writeText(code);
          }}>
            Export Code
          </button>
          <button className="run-btn">Run Model</button>
        </div>
      </div>
      
      <div className="editor-content">
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
                      <div
                        key={block.id}
                        className="pytorch-block"
                        draggable
                        onDragStart={(e) => handleDragStart(e, block)}
                        style={{ 
                          backgroundColor: block.color,
                          borderColor: block.color
                        }}
                      >
                        <div className="block-icon">{block.icon}</div>
                        <div className="block-label">{block.label}</div>
                        {Object.keys(block.config).length > 0 && (
                          <div className="block-params">
                            <div className="param-count">
                              {Object.keys(block.config).length} params
                            </div>
                          </div>
                        )}
                      </div>
                    ))
                  }
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div 
          className="canvas-container"
          ref={canvasRef}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <div className="canvas-background">
            <div className="canvas-grid"></div>
            {nodes.map(node => (
              <div
                key={node.id}
                className="canvas-node"
                style={{
                  left: node.x,
                  top: node.y,
                  borderColor: node.color
                }}
                onClick={() => setShowConfig(showConfig === node.id ? null : node.id)}
              >
                <div 
                  className="node-header"
                  style={{ backgroundColor: node.color }}
                >
                  <span className="node-icon">{node.icon}</span>
                  <span className="node-title">{node.label}</span>
                  {Object.keys(node.config).length > 0 && (
                    <button className="config-toggle">⚙️</button>
                  )}
                </div>
                
                {showConfig === node.id && Object.keys(node.config).length > 0 && (
                  <div className="node-config">
                    {Object.entries(node.config).map(([key, value]) => (
                      <div key={key} className="config-item">
                        <span className="config-key">{key}:</span>
                        <span className="config-value">{String(value)}</span>
                      </div>
                    ))}
                  </div>
                )}
                
                {/* Connection points */}
                {node.type !== 'input' && (
                  <div className="input-handle"></div>
                )}
                {node.type !== 'output' && !node.type.includes('loss') && node.type !== 'adam' && node.type !== 'sgd' && (
                  <div className="output-handle"></div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}