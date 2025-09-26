import React, { useState, useRef, useCallback, useEffect } from 'react';

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

interface DragState {
  isDragging: boolean;
  dragType: 'node' | 'connection';
  nodeId?: string;
  connectionStart?: {
    nodeId: string;
    port: string;
    type: 'input' | 'output';
    x: number;
    y: number;
  };
  currentPos?: { x: number; y: number };
  offset?: { x: number; y: number };
}

const PYTORCH_MODULES = {
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
      padding: { type: 'number', default: 0, min: 0, max: 50, description: 'Zero-padding added to all four sides of the input' }
    }
  },

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

  'BatchNorm2d': {
    category: 'Normalization',
    color: '#607D8B',
    icon: '🔄',
    inputs: ['input'],
    outputs: ['output'],
    params: {
      num_features: { type: 'number', default: 64, min: 1, max: 10000, description: 'Number of features or channels C of the input' },
      eps: { type: 'number', default: 1e-5, min: 1e-10, max: 1e-1, step: 1e-6, description: 'A value added to the denominator for numerical stability' },
      momentum: { type: 'number', default: 0.1, min: 0, max: 1, step: 0.01, description: 'The value used for the running_mean and running_var computation' }
    }
  },

  'Dropout': {
    category: 'Regularization',
    color: '#795548',
    icon: '❌',
    inputs: ['input'],
    outputs: ['output'],
    params: {
      p: { type: 'number', default: 0.5, min: 0, max: 1, step: 0.01, description: 'Probability of an element to be zeroed' }
    }
  },

  'CrossEntropyLoss': {
    category: 'Loss Functions',
    color: '#E91E63',
    icon: '🎯',
    inputs: ['input', 'target'],
    outputs: ['loss'],
    params: {
      reduction: { type: 'select', options: ['none', 'mean', 'sum'], default: 'mean', description: 'Specifies the reduction to apply to the output' }
    }
  }
};

const categories = [...new Set(Object.values(PYTORCH_MODULES).map(module => module.category))];

export function FinalVisualEditor() {
  const [selectedProject, setSelectedProject] = useState<string>('');
  const [isProjectCreated, setIsProjectCreated] = useState(false);
  const [nodes, setNodes] = useState<BlockNode[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [selectedNode, setSelectedNode] = useState<BlockNode | null>(null);
  const [dragState, setDragState] = useState<DragState>({ isDragging: false, dragType: 'node' });
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sidebarMode, setSidebarMode] = useState<'properties' | 'code'>('properties');
  const [generatedCode, setGeneratedCode] = useState<string>('');
  const [highlightedPorts, setHighlightedPorts] = useState<Set<string>>(new Set());
  
  const canvasRef = useRef<HTMLDivElement>(null);

  const handleCreateProject = () => {
    if (selectedProject.trim()) {
      setIsProjectCreated(true);
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
      updateGeneratedCode([inputNode], []);
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

    const newNodes = [...nodes, newNode];
    setNodes(newNodes);
    updateGeneratedCode(newNodes, connections);
  }, [nodes, connections]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const target = e.target as Element;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    // Check if clicking on a port
    if (target.classList.contains('port')) {
      const nodeElement = target.closest('.canvas-node');
      const nodeId = nodeElement?.getAttribute('data-node-id');
      const portType = target.classList.contains('input-port') ? 'input' : 'output';
      const portName = target.getAttribute('data-port') || '';
      
      if (nodeId) {
        setDragState({
          isDragging: true,
          dragType: 'connection',
          connectionStart: {
            nodeId,
            port: portName,
            type: portType,
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
          },
          currentPos: { x: e.clientX - rect.left, y: e.clientY - rect.top }
        });
        
        // Highlight compatible ports
        highlightCompatiblePorts(nodeId, portType);
      }
      return;
    }

    // Check if clicking on a node
    if (target.closest('.canvas-node')) {
      const nodeElement = target.closest('.canvas-node');
      const nodeId = nodeElement?.getAttribute('data-node-id');
      const node = nodes.find(n => n.id === nodeId);
      
      if (node) {
        setSelectedNode(node);
        setDragState({
          isDragging: true,
          dragType: 'node',
          nodeId: node.id,
          offset: {
            x: e.clientX - rect.left - node.x,
            y: e.clientY - rect.top - node.y
          }
        });
      }
    } else {
      // Clicking on empty space
      setSelectedNode(null);
    }
  }, [nodes]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragState.isDragging || !canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const currentPos = { x: e.clientX - rect.left, y: e.clientY - rect.top };

    if (dragState.dragType === 'node' && dragState.nodeId && dragState.offset) {
      const newX = Math.max(0, currentPos.x - dragState.offset.x);
      const newY = Math.max(0, currentPos.y - dragState.offset.y);
      
      const newNodes = nodes.map(node => 
        node.id === dragState.nodeId 
          ? { ...node, x: newX, y: newY }
          : node
      );
      setNodes(newNodes);
    } else if (dragState.dragType === 'connection') {
      setDragState(prev => ({ ...prev, currentPos }));
    }
  }, [dragState, nodes]);

  const handleMouseUp = useCallback((e: React.MouseEvent) => {
    if (!dragState.isDragging) return;

    if (dragState.dragType === 'connection' && dragState.connectionStart) {
      const target = e.target as Element;
      
      if (target.classList.contains('port')) {
        const nodeElement = target.closest('.canvas-node');
        const targetNodeId = nodeElement?.getAttribute('data-node-id');
        const targetPortType = target.classList.contains('input-port') ? 'input' : 'output';
        const targetPortName = target.getAttribute('data-port') || '';
        
        if (targetNodeId && 
            targetNodeId !== dragState.connectionStart.nodeId &&
            targetPortType !== dragState.connectionStart.type) {
          
          const newConnection: Connection = {
            id: `conn-${Date.now()}`,
            fromNodeId: dragState.connectionStart.type === 'output' ? dragState.connectionStart.nodeId : targetNodeId,
            toNodeId: dragState.connectionStart.type === 'input' ? dragState.connectionStart.nodeId : targetNodeId,
            fromPort: dragState.connectionStart.type === 'output' ? dragState.connectionStart.port : targetPortName,
            toPort: dragState.connectionStart.type === 'input' ? dragState.connectionStart.port : targetPortName
          };
          
          const newConnections = [...connections, newConnection];
          setConnections(newConnections);
          updateGeneratedCode(nodes, newConnections);
        }
      }
    }

    setDragState({ isDragging: false, dragType: 'node' });
    setHighlightedPorts(new Set());
  }, [dragState, connections, nodes]);

  const highlightCompatiblePorts = (nodeId: string, portType: 'input' | 'output') => {
    const compatiblePorts = new Set<string>();
    const oppositeType = portType === 'input' ? 'output' : 'input';
    
    nodes.forEach(node => {
      if (node.id !== nodeId) {
        const ports = oppositeType === 'input' ? node.inputs : node.outputs;
        ports.forEach(port => {
          compatiblePorts.add(`${node.id}-${port}-${oppositeType}`);
        });
      }
    });
    
    setHighlightedPorts(compatiblePorts);
  };

  const getPortPosition = (node: BlockNode, port: string, type: 'input' | 'output') => {
    const ports = type === 'input' ? node.inputs : node.outputs;
    const index = ports.indexOf(port);
    const portWidth = 180 / (ports.length + 1);
    const x = node.x + (index + 1) * portWidth;
    const y = node.y + (type === 'input' ? 0 : 60);
    return { x, y };
  };

  const updateNodeConfig = (nodeId: string, key: string, value: any) => {
    const newNodes = nodes.map(node => 
      node.id === nodeId 
        ? { ...node, config: { ...node.config, [key]: value } }
        : node
    );
    setNodes(newNodes);
    updateGeneratedCode(newNodes, connections);
  };

  const updateGeneratedCode = (nodeList: BlockNode[], connectionList: Connection[]) => {
    let code = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n";
    code += "class GeneratedModel(nn.Module):\n";
    code += "    def __init__(self):\n";
    code += "        super(GeneratedModel, self).__init__()\n";
    
    nodeList.forEach((node, index) => {
      if (node.type !== 'Input') {
        const layerName = `layer_${index}`;
        code += `        self.${layerName} = ${getPytorchCode(node)}\n`;
      }
    });
    
    code += "\n    def forward(self, x):\n";
    code += "        # Forward pass implementation\n";
    
    const sortedNodes = nodeList.filter(node => node.type !== 'Input').sort((a, b) => a.y - b.y);
    sortedNodes.forEach((node, index) => {
      const layerName = `layer_${nodeList.indexOf(node)}`;
      code += `        x = self.${layerName}(x)\n`;
    });
    
    code += "        return x\n\n";
    code += "# Create model instance\nmodel = GeneratedModel()\nprint(model)";
    
    setGeneratedCode(code);
  };

  const getPytorchCode = (node: BlockNode) => {
    const config = node.config;
    switch (node.type) {
      case 'Linear':
        return `nn.Linear(${config.in_features}, ${config.out_features}, bias=${config.bias})`;
      case 'Conv2d':
        return `nn.Conv2d(${config.in_channels}, ${config.out_channels}, ${config.kernel_size}, stride=${config.stride}, padding=${config.padding})`;
      case 'ReLU':
        return `nn.ReLU(inplace=${config.inplace})`;
      case 'BatchNorm2d':
        return `nn.BatchNorm2d(${config.num_features})`;
      case 'Dropout':
        return `nn.Dropout(p=${config.p})`;
      case 'CrossEntropyLoss':
        return `nn.CrossEntropyLoss(reduction='${config.reduction}')`;
      default:
        return 'nn.Identity()';
    }
  };

  useEffect(() => {
    updateGeneratedCode(nodes, connections);
  }, [nodes, connections]);

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
    <div className="final-visual-editor">
      <div className="editor-header">
        <h2>Project: {selectedProject}</h2>
        <div className="header-actions">
          <button className="export-btn" onClick={() => {
            navigator.clipboard.writeText(generatedCode);
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
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        >
          <svg className="connection-svg">
            {connections.map(conn => {
              const fromNode = nodes.find(n => n.id === conn.fromNodeId);
              const toNode = nodes.find(n => n.id === conn.toNodeId);
              if (!fromNode || !toNode) return null;
              
              const fromPos = getPortPosition(fromNode, conn.fromPort, 'output');
              const toPos = getPortPosition(toNode, conn.toPort, 'input');
              
              // Bezier curve for smooth connections
              const midX = (fromPos.x + toPos.x) / 2;
              const path = `M ${fromPos.x} ${fromPos.y} C ${midX} ${fromPos.y + 30}, ${midX} ${toPos.y - 30}, ${toPos.x} ${toPos.y}`;
              
              return (
                <path
                  key={conn.id}
                  d={path}
                  stroke="#666"
                  strokeWidth="3"
                  fill="none"
                  markerEnd="url(#arrowhead)"
                />
              );
            })}
            
            {/* Drag connection preview */}
            {dragState.isDragging && dragState.dragType === 'connection' && dragState.connectionStart && dragState.currentPos && (
              <line
                x1={dragState.connectionStart.x}
                y1={dragState.connectionStart.y}
                x2={dragState.currentPos.x}
                y2={dragState.currentPos.y}
                stroke="#2196F3"
                strokeWidth="3"
                strokeDasharray="8,4"
                markerEnd="url(#arrowhead-blue)"
              />
            )}
            
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
              </marker>
              <marker id="arrowhead-blue" markerWidth="10" markerHeight="7" 
                refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#2196F3" />
              </marker>
            </defs>
          </svg>
          
          {nodes.map(node => (
            <div
              key={node.id}
              className={`canvas-node ${selectedNode?.id === node.id ? 'selected' : ''}`}
              data-node-id={node.id}
              style={{
                left: node.x,
                top: node.y,
                borderColor: node.color
              }}
            >
              {/* Input ports */}
              {node.inputs.map((input, index) => (
                <div
                  key={input}
                  className={`input-port port ${highlightedPorts.has(`${node.id}-${input}-input`) ? 'highlighted' : ''}`}
                  data-port={input}
                  style={{ left: `${(index + 1) * (180 / (node.inputs.length + 1)) - 6}px` }}
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
                  className={`output-port port ${highlightedPorts.has(`${node.id}-${output}-output`) ? 'highlighted' : ''}`}
                  data-port={output}
                  style={{ left: `${(index + 1) * (180 / (node.outputs.length + 1)) - 6}px` }}
                />
              ))}
            </div>
          ))}
        </div>
        
        {/* Right Sidebar */}
        {sidebarOpen && (
          <div className="properties-panel">
            <div className="panel-header">
              <div className="panel-tabs">
                <button 
                  className={sidebarMode === 'properties' ? 'active' : ''}
                  onClick={() => setSidebarMode('properties')}
                >
                  Properties
                </button>
                <button 
                  className={sidebarMode === 'code' ? 'active' : ''}
                  onClick={() => setSidebarMode('code')}
                >
                  Code
                </button>
              </div>
              <button 
                className="close-sidebar-btn"
                onClick={() => setSidebarOpen(false)}
              >
                ✕
              </button>
            </div>
            
            <div className="panel-content">
              {sidebarMode === 'properties' ? (
                selectedNode ? (
                  <div className="node-properties">
                    <h4>{selectedNode.type}</h4>
                    
                    {PYTORCH_MODULES[selectedNode.type as keyof typeof PYTORCH_MODULES] && (
                      <div className="parameters">
                        <h5>Parameters</h5>
                        {Object.entries(PYTORCH_MODULES[selectedNode.type as keyof typeof PYTORCH_MODULES].params).map(([key, param]: [string, any]) => (
                          <div key={key} className="parameter">
                            <label htmlFor={`param-${key}`}>{key}:</label>
                            <div className="param-description">{param.description}</div>
                            {param.type === 'number' ? (
                              <input
                                id={`param-${key}`}
                                type="number"
                                value={selectedNode.config[key] ?? param.default}
                                min={param.min}
                                max={param.max}
                                step={param.step || 1}
                                onChange={(e) => {
                                  const value = parseFloat(e.target.value);
                                  updateNodeConfig(selectedNode.id, key, isNaN(value) ? param.default : value);
                                }}
                              />
                            ) : param.type === 'boolean' ? (
                              <label className="checkbox-label">
                                <input
                                  id={`param-${key}`}
                                  type="checkbox"
                                  checked={selectedNode.config[key] ?? param.default}
                                  onChange={(e) => updateNodeConfig(selectedNode.id, key, e.target.checked)}
                                />
                                <span className="checkmark"></span>
                              </label>
                            ) : param.type === 'select' ? (
                              <select
                                id={`param-${key}`}
                                value={selectedNode.config[key] ?? param.default}
                                onChange={(e) => updateNodeConfig(selectedNode.id, key, e.target.value)}
                              >
                                {param.options.map((option: string) => (
                                  <option key={option} value={option}>{option}</option>
                                ))}
                              </select>
                            ) : (
                              <input
                                id={`param-${key}`}
                                type="text"
                                value={selectedNode.config[key] ?? param.default}
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
                )
              ) : (
                <div className="code-editor">
                  <div className="code-header">
                    <h5>Generated PyTorch Code</h5>
                    <button 
                      onClick={() => navigator.clipboard.writeText(generatedCode)}
                      className="copy-code-btn"
                    >
                      Copy
                    </button>
                  </div>
                  <textarea 
                    className="code-content editable"
                    value={generatedCode}
                    onChange={(e) => setGeneratedCode(e.target.value)}
                    spellCheck={false}
                  />
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Sidebar Toggle Button */}
        {!sidebarOpen && (
          <button 
            className="open-sidebar-btn"
            onClick={() => setSidebarOpen(true)}
          >
            ⚙️
          </button>
        )}
      </div>
    </div>
  );
}