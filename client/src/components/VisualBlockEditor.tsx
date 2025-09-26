import React, { useCallback, useState } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  Node,
  NodeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { PyTorchBlockPalette } from './PyTorchBlockPalette';
import { PyTorchNode } from './PyTorchNode';

const nodeTypes: NodeTypes = {
  pytorchNode: PyTorchNode,
};

const initialNodes: Node[] = [
  {
    id: '1',
    type: 'pytorchNode',
    position: { x: 250, y: 50 },
    data: { 
      label: 'Input Layer',
      moduleType: 'input',
      config: { shape: [1, 784] }
    },
  },
];

const initialEdges: Edge[] = [];

export function VisualBlockEditor() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedProject, setSelectedProject] = useState<string>('');
  const [isProjectCreated, setIsProjectCreated] = useState(false);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const blockData = event.dataTransfer.getData('application/pytorch-block');
      
      if (!blockData) return;

      const { moduleType, label, config } = JSON.parse(blockData);
      const position = {
        x: event.clientX - 200,
        y: event.clientY - 100,
      };

      const newNode: Node = {
        id: `${Date.now()}`,
        type: 'pytorchNode',
        position,
        data: { label, moduleType, config },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [setNodes]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const handleCreateProject = () => {
    if (selectedProject.trim()) {
      setIsProjectCreated(true);
    }
  };

  const generatePythonCode = () => {
    let code = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n";
    code += "class GeneratedModel(nn.Module):\n";
    code += "    def __init__(self):\n";
    code += "        super(GeneratedModel, self).__init__()\n";
    
    // Add layers based on nodes
    nodes.forEach((node, index) => {
      if (node.data.moduleType !== 'input') {
        const layerName = `layer_${index}`;
        code += `        self.${layerName} = ${getPytorchCode(node.data)}\n`;
      }
    });
    
    code += "\n    def forward(self, x):\n";
    code += "        # Forward pass implementation\n";
    code += "        return x\n\n";
    code += "# Create model instance\nmodel = GeneratedModel()\nprint(model)";
    
    return code;
  };

  const getPytorchCode = (data: any) => {
    switch (data.moduleType) {
      case 'linear':
        return `nn.Linear(${data.config?.input_features || 784}, ${data.config?.output_features || 128})`;
      case 'conv2d':
        return `nn.Conv2d(${data.config?.in_channels || 1}, ${data.config?.out_channels || 32}, ${data.config?.kernel_size || 3})`;
      case 'relu':
        return 'nn.ReLU()';
      case 'sigmoid':
        return 'nn.Sigmoid()';
      case 'dropout':
        return `nn.Dropout(${data.config?.p || 0.5})`;
      case 'batchnorm':
        return `nn.BatchNorm1d(${data.config?.num_features || 128})`;
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
        <PyTorchBlockPalette />
        
        <div className="flow-container">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onDrop={onDrop}
            onDragOver={onDragOver}
            nodeTypes={nodeTypes}
            fitView
            className="react-flow-canvas"
          >
            <Controls />
            <MiniMap />
            <Background variant="dots" gap={12} size={1} />
          </ReactFlow>
        </div>
      </div>
    </div>
  );
}