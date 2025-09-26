import React, { useState, useRef, useCallback } from 'react';

interface BlockNode {
  id: string;
  type: string;
  label: string;
  x: number;
  y: number;
  config: any;
  color: string;
  icon: string;
  inputs: string[];
  outputs: string[];
}

interface Connection {
  id: string;
  fromNodeId: string;
  toNodeId: string;
  fromPort: string;
  toPort: string;
}

const PYTORCH_MODULES = {
  // Linear Layers
  'Linear': {
    category: 'Linear Layers',
    color: '#2196F3',
    icon: '🔗',
    inputs: ['input'],
    outputs: ['output'],
    params: {
      in_features: { type: 'number', default: 784, min: 1, max: 10000, description: 'Size of each input sample' },
      out_features: { type: 'number', default: 128, min: 1, max: 10000, description: 'Size of each output sample' },
      bias: { type: 'boolean', default: true, description: 'If set to False, the layer will not learn an additive bias' }
    }
  },
  'Bilinear': {
    category: 'Linear Layers',
    color: '#2196F3',
    icon: '🔗',
    inputs: ['input1', 'input2'],
    outputs: ['output'],
    params: {
      in1_features: { type: 'number', default: 128, min: 1, max: 10000, description: 'Size of first input sample' },
      in2_features: { type: 'number', default: 128, min: 1, max: 10000, description: 'Size of second input sample' },
      out_features: { type: 'number', default: 64, min: 1, max: 10000, description: 'Size of each output sample' },
      bias: { type: 'boolean', default: true, description: 'If set to False, the layer will not learn an additive bias' }
    }
  },

  // Convolution Layers
  'Conv1d': {
    category: 'Convolution',
    color: '#FF9800',
    icon: '📊',
    inputs: ['input'],
    outputs: ['output'],
    params: {
      in_channels: { type: 'number', default: 1, min: 1, max: 1000, description: 'Number of channels in the input signal' },
      out_channels: { type: 'number', default: 32, min: 1, max: 1000, description: 'Number of channels produced by the convolution' },
      kernel_size: { type: 'number', default: 3, min: 1, max: 101, description: 'Size of the convolving kernel' },
      stride: { type: 'number', default: 1, min: 1, max: 50, description: 'Stride of the convolution' },
      padding: { type: 'number', default: 0, min: 0, max: 50, description: 'Zero-padding added to both sides of the input' },
      dilation: { type: 'number', default: 1, min: 1, max: 50, description: 'Spacing between kernel elements' },
      groups: { type: 'number', default: 1, min: 1, max: 1000, description: 'Number of blocked connections' },
      bias: { type: 'boolean', default: true, description: 'If True, adds a learnable bias to the output' }
    }
  },
  'Conv2d': {
    category: 'Convolution',
    color: '#FF9800',
    icon: '🖼️',
    inputs: ['input'],
    outputs: ['output'],
    params: {
      in_channels: { type: 'number', default: 3, min: 1, max: 1000, description: 'Number of channels in the input image' },
      out_channels: { type: 'number', default: 64, min: 1, max: 1000, description: 'Number of channels produced by the convolution' },
      kernel_size: { type: 'number', default: 3, min: 1, max: 101, description: 'Size of the convolving kernel' },
      stride: { type: 'number', default: 1, min: 1, max: 50, description: 'Stride of the convolution' },
      padding: { type: 'number', default: 0, min: 0, max: 50, description: 'Zero-padding added to all four sides of the input' },
      dilation: { type: 'number', default: 1, min: 1, max: 50, description: 'Spacing between kernel elements' },
      groups: { type: 'number', default: 1, min: 1, max: 1000, description: 'Number of blocked connections' },
      bias: { type: 'boolean', default: true, description: 'If True, adds a learnable bias to the output' }
    }
  },

  // Activation Functions
  'ReLU': {
    category: 'Activation',
    color: '#F44336',
    icon: '⚡',
    inputs: ['input'],
    outputs: ['output'],
    params: {
      inplace: { type: 'boolean', default: false, description: 'Can optionally do the operation in-place' }
    }
  },
  'LeakyReLU': {
    category: 'Activation',
    color: '#F44336',
    icon: '⚡',
    inputs: ['input'],
    outputs: ['output'],
    params: {
      negative_slope: { type: 'number', default: 0.01, min: 0, max: 1, step: 0.001, description: 'Controls the angle of the negative slope' },
      inplace: { type: 'boolean', default: false, description: 'Can optionally do the operation in-place' }
    }
  },
  'Sigmoid': {
    category: 'Activation',
    color: '#F44336',
    icon: '〰️',
    inputs: ['input'],
    outputs: ['output'],
    params: {}
  },
  'Tanh': {
    category: 'Activation',
    color: '#F44336',
    icon: '📈',
    inputs: ['input'],
    outputs: ['output'],
    params: {}
  },
  'Softmax': {
    category: 'Activation',
    color: '#F44336',
    icon: '🎯',
    inputs: ['input'],
    outputs: ['output'],
    params: {
      dim: { type: 'number', default: 1, description: 'A dimension along which Softmax will be computed' }
    }
  },

  // Pooling Layers
  'MaxPool1d': {
    category: 'Pooling',
    color: '#9C27B0',
    icon: '⬇️',
    inputs: ['input'],
    outputs: ['output'],
    params: {
      kernel_size: { type: 'number', default: 2, min: 1, max: 50, description: 'The size of the window to take a max over' },
      stride: { type: 'number', default: null, min: 1, max: 50, description: 'The stride of the window' },
      padding: { type: 'number', default: 0, min: 0, max: 25, description: 'Implicit zero padding to be added on both sides' },
      dilation: { type: 'number', default: 1, min: 1, max: 50, description: 'A parameter that controls the stride of elements in the window' },
      return_indices: { type: 'boolean', default: false, description: 'If True, will return the max indices along with the outputs' },
      ceil_mode: { type: 'boolean', default: false, description: 'When True, will use ceil instead of floor to compute the output shape' }
    }
  },
  'MaxPool2d': {
    category: 'Pooling',
    color: '#9C27B0',
    icon: '⬇️',
    inputs: ['input'],
    outputs: ['output'],
    params: {
      kernel_size: { type: 'number', default: 2, min: 1, max: 50, description: 'The size of the window to take a max over' },
      stride: { type: 'number', default: null, min: 1, max: 50, description: 'The stride of the window' },
      padding: { type: 'number', default: 0, min: 0, max: 25, description: 'Implicit zero padding to be added on all four sides' },
      dilation: { type: 'number', default: 1, min: 1, max: 50, description: 'A parameter that controls the stride of elements in the window' },
      return_indices: { type: 'boolean', default: false, description: 'If True, will return the max indices along with the outputs' },
      ceil_mode: { type: 'boolean', default: false, description: 'When True, will use ceil instead of floor to compute the output shape' }
    }
  },
  'AvgPool2d': {
    category: 'Pooling',
    color: '#9C27B0',
    icon: '📉',
    inputs: ['input'],
    outputs: ['output'],
    params: {
      kernel_size: { type: 'number', default: 2, min: 1, max: 50, description: 'The size of the window' },
      stride: { type: 'number', default: null, min: 1, max: 50, description: 'The stride of the window' },
      padding: { type: 'number', default: 0, min: 0, max: 25, description: 'Implicit zero padding to be added on all four sides' },
      ceil_mode: { type: 'boolean', default: false, description: 'When True, will use ceil instead of floor to compute the output shape' },
      count_include_pad: { type: 'boolean', default: true, description: 'When True, will include the zero-padding in the averaging calculation' },
      divisor_override: { type: 'number', default: null, description: 'If specified, it will be used as divisor, otherwise size of the pooling region will be used' }
    }
  },

  // Normalization
  'BatchNorm1d': {
    category: 'Normalization',
    color: '#607D8B',
    icon: '🔄',
    inputs: ['input'],
    outputs: ['output'],
    params: {
      num_features: { type: 'number', default: 128, min: 1, max: 10000, description: 'Number of features or channels C of the input' },
      eps: { type: 'number', default: 1e-5, min: 1e-10, max: 1e-1, step: 1e-6, description: 'A value added to the denominator for numerical stability' },
      momentum: { type: 'number', default: 0.1, min: 0, max: 1, step: 0.01, description: 'The value used for the running_mean and running_var computation' },
      affine: { type: 'boolean', default: true, description: 'A boolean value that when set to True, this module has learnable affine parameters' },
      track_running_stats: { type: 'boolean', default: true, description: 'A boolean value that when set to True, this module tracks the running mean and variance' }
    }
  },
  'BatchNorm2d': {
    category: 'Normalization',
    color: '#607D8B',
    icon: '🔄',
    inputs: ['input'],
    outputs: ['output'],
    params: {
      num_features: { type: 'number', default: 64, min: 1, max: 10000, description: 'Number of features or channels C of the input' },
      eps: { type: 'number', default: 1e-5, min: 1e-10, max: 1e-1, step: 1e-6, description: 'A value added to the denominator for numerical stability' },
      momentum: { type: 'number', default: 0.1, min: 0, max: 1, step: 0.01, description: 'The value used for the running_mean and running_var computation' },
      affine: { type: 'boolean', default: true, description: 'A boolean value that when set to True, this module has learnable affine parameters' },
      track_running_stats: { type: 'boolean', default: true, description: 'A boolean value that when set to True, this module tracks the running mean and variance' }
    }
  },

  // Regularization
  'Dropout': {
    category: 'Regularization',
    color: '#795548',
    icon: '❌',
    inputs: ['input'],
    outputs: ['output'],
    params: {
      p: { type: 'number', default: 0.5, min: 0, max: 1, step: 0.01, description: 'Probability of an element to be zeroed' },
      inplace: { type: 'boolean', default: false, description: 'If set to True, will do this operation in-place' }
    }
  },

  // Loss Functions
  'CrossEntropyLoss': {
    category: 'Loss Functions',
    color: '#E91E63',
    icon: '🎯',
    inputs: ['input', 'target'],
    outputs: ['loss'],
    params: {
      weight: { type: 'tensor', default: null, description: 'A manual rescaling weight given to each class' },
      size_average: { type: 'boolean', default: null, description: 'Deprecated' },
      ignore_index: { type: 'number', default: -100, description: 'Specifies a target value that is ignored and does not contribute to the input gradient' },
      reduce: { type: 'boolean', default: null, description: 'Deprecated' },
      reduction: { type: 'select', options: ['none', 'mean', 'sum'], default: 'mean', description: 'Specifies the reduction to apply to the output' },
      label_smoothing: { type: 'number', default: 0.0, min: 0, max: 1, step: 0.01, description: 'A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss' }
    }
  },
  'MSELoss': {
    category: 'Loss Functions', 
    color: '#E91E63',
    icon: '📏',
    inputs: ['input', 'target'],
    outputs: ['loss'],
    params: {
      size_average: { type: 'boolean', default: null, description: 'Deprecated' },
      reduce: { type: 'boolean', default: null, description: 'Deprecated' },
      reduction: { type: 'select', options: ['none', 'mean', 'sum'], default: 'mean', description: 'Specifies the reduction to apply to the output' }
    }
  }
};

const categories = [...new Set(Object.values(PYTORCH_MODULES).map(module => module.category))];

export function AdvancedVisualEditor() {
  const [selectedProject, setSelectedProject] = useState<string>('');
  const [isProjectCreated, setIsProjectCreated] = useState(false);
  const [nodes, setNodes] = useState<BlockNode[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [selectedNode, setSelectedNode] = useState<BlockNode | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [connecting, setConnecting] = useState<{ nodeId: string, port: string, type: 'input' | 'output' } | null>(null);
  
  const canvasRef = useRef<HTMLDivElement>(null);

  const handleCreateProject = () => {
    if (selectedProject.trim()) {
      setIsProjectCreated(true);
      // Add initial input node
      const inputNode: BlockNode = {
        id: `input-${Date.now()}`,
        type: 'Input',
        label: 'Input',
        x: 100,
        y: 100,
        config: { shape: [1, 784] },
        color: '#4CAF50',
        icon: '📥',
        inputs: [],
        outputs: ['output']
      };
      setNodes([inputNode]);
    }
  };

  const handleDragStart = (e: React.DragEvent, moduleType: string) => {
    const moduleData = PYTORCH_MODULES[moduleType as keyof typeof PYTORCH_MODULES];
    e.dataTransfer.setData('application/pytorch-block', JSON.stringify({
      type: moduleType,
      ...moduleData
    }));
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const blockData = JSON.parse(e.dataTransfer.getData('application/pytorch-block'));
    
    if (!canvasRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const config: any = {};
    Object.entries(blockData.params || {}).forEach(([key, param]: [string, any]) => {
      config[key] = param.default;
    });

    const newNode: BlockNode = {
      id: `${blockData.type}-${Date.now()}`,
      type: blockData.type,
      label: blockData.type,
      x: Math.max(0, x - 90),
      y: Math.max(0, y - 40),
      config,
      color: blockData.color,
      icon: blockData.icon,
      inputs: blockData.inputs || [],
      outputs: blockData.outputs || []
    };

    setNodes(prev => [...prev, newNode]);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleNodeMouseDown = (e: React.MouseEvent, node: BlockNode) => {
    if (e.target !== e.currentTarget && (e.target as Element).classList.contains('port')) {
      return; // Don't drag if clicking on a port
    }
    
    setIsDragging(true);
    setSelectedNode(node);
    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect) {
      setDragOffset({
        x: e.clientX - rect.left - node.x,
        y: e.clientY - rect.top - node.y
      });
    }
  };

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging || !selectedNode || !canvasRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const newX = Math.max(0, e.clientX - rect.left - dragOffset.x);
    const newY = Math.max(0, e.clientY - rect.top - dragOffset.y);
    
    setNodes(prev => prev.map(node => 
      node.id === selectedNode.id 
        ? { ...node, x: newX, y: newY }
        : node
    ));
  }, [isDragging, selectedNode, dragOffset]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handlePortClick = (nodeId: string, port: string, type: 'input' | 'output') => {
    if (connecting) {
      if (connecting.type !== type && connecting.nodeId !== nodeId) {
        // Create connection
        const newConnection: Connection = {
          id: `conn-${Date.now()}`,
          fromNodeId: connecting.type === 'output' ? connecting.nodeId : nodeId,
          toNodeId: connecting.type === 'input' ? connecting.nodeId : nodeId,
          fromPort: connecting.type === 'output' ? connecting.port : port,
          toPort: connecting.type === 'input' ? connecting.port : port
        };
        setConnections(prev => [...prev, newConnection]);
      }
      setConnecting(null);
    } else {
      setConnecting({ nodeId, port, type });
    }
  };

  const updateNodeConfig = (nodeId: string, key: string, value: any) => {
    setNodes(prev => prev.map(node => 
      node.id === nodeId 
        ? { ...node, config: { ...node.config, [key]: value } }
        : node
    ));
  };

  const generatePythonCode = () => {
    let code = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n";
    code += "class GeneratedModel(nn.Module):\n";
    code += "    def __init__(self):\n";
    code += "        super(GeneratedModel, self).__init__()\n";
    
    nodes.forEach((node, index) => {
      if (node.type !== 'Input') {
        const layerName = `layer_${index}`;
        code += `        self.${layerName} = ${getPytorchCode(node)}\n`;
      }
    });
    
    code += "\n    def forward(self, x):\n";
    code += "        # Forward pass implementation based on connections\n";
    code += "        return x\n\n";
    code += "# Create model instance\nmodel = GeneratedModel()\nprint(model)";
    
    return code;
  };

  const getPytorchCode = (node: BlockNode) => {
    const config = node.config;
    switch (node.type) {
      case 'Linear':
        return `nn.Linear(${config.in_features}, ${config.out_features}, bias=${config.bias})`;
      case 'Conv2d':
        return `nn.Conv2d(${config.in_channels}, ${config.out_channels}, ${config.kernel_size}, stride=${config.stride}, padding=${config.padding}, bias=${config.bias})`;
      case 'ReLU':
        return `nn.ReLU(inplace=${config.inplace})`;
      case 'Dropout':
        return `nn.Dropout(${config.p}, inplace=${config.inplace})`;
      case 'BatchNorm2d':
        return `nn.BatchNorm2d(${config.num_features}, eps=${config.eps}, momentum=${config.momentum})`;
      case 'MaxPool2d':
        return `nn.MaxPool2d(${config.kernel_size}, stride=${config.stride}, padding=${config.padding})`;
      case 'CrossEntropyLoss':
        return `nn.CrossEntropyLoss(reduction='${config.reduction}')`;
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
    <div className="advanced-visual-editor">
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
        {/* Left Palette */}
        <div className="pytorch-block-palette">
          <div className="palette-header">
            <h3>PyTorch Modules</h3>
            <p>Drag modules to canvas</p>
          </div>
          
          <div className="palette-content">
            {categories.map(category => (
              <div key={category} className="block-category">
                <h4 className="category-header">{category}</h4>
                <div className="category-blocks">
                  {Object.entries(PYTORCH_MODULES)
                    .filter(([_, module]) => module.category === category)
                    .map(([moduleType, module]) => (
                      <div
                        key={moduleType}
                        className="pytorch-block"
                        draggable
                        onDragStart={(e) => handleDragStart(e, moduleType)}
                        style={{ 
                          backgroundColor: module.color,
                          borderColor: module.color
                        }}
                      >
                        <div className="block-icon">{module.icon}</div>
                        <div className="block-label">{moduleType}</div>
                        {Object.keys(module.params).length > 0 && (
                          <div className="block-params">
                            <div className="param-count">
                              {Object.keys(module.params).length} params
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
        
        {/* Center Canvas */}
        <div 
          className="canvas-container"
          ref={canvasRef}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onClick={() => setConnecting(null)}
        >
          <svg className="connection-svg">
            {connections.map(conn => {
              const fromNode = nodes.find(n => n.id === conn.fromNodeId);
              const toNode = nodes.find(n => n.id === conn.toNodeId);
              if (!fromNode || !toNode) return null;
              
              const fromX = fromNode.x + 90;
              const fromY = fromNode.y + 60;
              const toX = toNode.x + 90;
              const toY = toNode.y + 20;
              
              return (
                <line
                  key={conn.id}
                  x1={fromX}
                  y1={fromY}
                  x2={toX}
                  y2={toY}
                  stroke="#666"
                  strokeWidth="2"
                  markerEnd="url(#arrowhead)"
                />
              );
            })}
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
              </marker>
            </defs>
          </svg>
          
          {nodes.map(node => (
            <div
              key={node.id}
              className={`canvas-node ${selectedNode?.id === node.id ? 'selected' : ''}`}
              style={{
                left: node.x,
                top: node.y,
                borderColor: node.color
              }}
              onMouseDown={(e) => handleNodeMouseDown(e, node)}
              onClick={(e) => {
                e.stopPropagation();
                setSelectedNode(node);
              }}
            >
              {/* Input ports */}
              {node.inputs.map((input, index) => (
                <div
                  key={input}
                  className="input-port port"
                  style={{ left: `${(index + 1) * (180 / (node.inputs.length + 1)) - 5}px` }}
                  onClick={(e) => {
                    e.stopPropagation();
                    handlePortClick(node.id, input, 'input');
                  }}
                />
              ))}
              
              <div 
                className="node-header"
                style={{ backgroundColor: node.color }}
              >
                <span className="node-icon">{node.icon}</span>
                <span className="node-title">{node.label}</span>
              </div>
              
              {/* Output ports */}
              {node.outputs.map((output, index) => (
                <div
                  key={output}
                  className="output-port port"
                  style={{ left: `${(index + 1) * (180 / (node.outputs.length + 1)) - 5}px` }}
                  onClick={(e) => {
                    e.stopPropagation();
                    handlePortClick(node.id, output, 'output');
                  }}
                />
              ))}
            </div>
          ))}
        </div>
        
        {/* Right Properties Panel */}
        <div className="properties-panel">
          <div className="panel-header">
            <h3>Properties</h3>
          </div>
          
          <div className="panel-content">
            {selectedNode ? (
              <div className="node-properties">
                <h4>{selectedNode.type}</h4>
                <div className="property-group">
                  <label>Node ID:</label>
                  <span className="node-id">{selectedNode.id}</span>
                </div>
                
                {PYTORCH_MODULES[selectedNode.type as keyof typeof PYTORCH_MODULES] && (
                  <div className="parameters">
                    <h5>Parameters</h5>
                    {Object.entries(PYTORCH_MODULES[selectedNode.type as keyof typeof PYTORCH_MODULES].params).map(([key, param]: [string, any]) => (
                      <div key={key} className="parameter">
                        <label>{key}:</label>
                        <div className="param-description">{param.description}</div>
                        {param.type === 'number' ? (
                          <input
                            type="number"
                            value={selectedNode.config[key] || param.default}
                            min={param.min}
                            max={param.max}
                            step={param.step}
                            onChange={(e) => updateNodeConfig(selectedNode.id, key, parseFloat(e.target.value))}
                          />
                        ) : param.type === 'boolean' ? (
                          <input
                            type="checkbox"
                            checked={selectedNode.config[key] ?? param.default}
                            onChange={(e) => updateNodeConfig(selectedNode.id, key, e.target.checked)}
                          />
                        ) : param.type === 'select' ? (
                          <select
                            value={selectedNode.config[key] || param.default}
                            onChange={(e) => updateNodeConfig(selectedNode.id, key, e.target.value)}
                          >
                            {param.options.map((option: string) => (
                              <option key={option} value={option}>{option}</option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type="text"
                            value={selectedNode.config[key] || param.default}
                            onChange={(e) => updateNodeConfig(selectedNode.id, key, e.target.value)}
                          />
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="no-selection">
                <p>Select a node to edit its properties</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}