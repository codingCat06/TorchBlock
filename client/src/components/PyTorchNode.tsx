import React, { memo, useState } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';

interface PyTorchNodeData {
  label: string;
  moduleType: string;
  config: any;
}

export const PyTorchNode = memo(({ data, selected }: NodeProps<PyTorchNodeData>) => {
  const [isConfigOpen, setIsConfigOpen] = useState(false);

  const getNodeColor = (moduleType: string) => {
    const colorMap: { [key: string]: string } = {
      'input': '#4CAF50',
      'output': '#4CAF50', 
      'linear': '#2196F3',
      'bilinear': '#2196F3',
      'conv1d': '#FF9800',
      'conv2d': '#FF9800',
      'conv3d': '#FF9800',
      'maxpool1d': '#9C27B0',
      'maxpool2d': '#9C27B0',
      'avgpool2d': '#9C27B0',
      'relu': '#F44336',
      'sigmoid': '#F44336',
      'tanh': '#F44336',
      'softmax': '#F44336',
      'leaky_relu': '#F44336',
      'batchnorm1d': '#607D8B',
      'batchnorm2d': '#607D8B',
      'layernorm': '#607D8B',
      'dropout': '#795548',
      'dropout2d': '#795548',
      'cross_entropy': '#E91E63',
      'mse_loss': '#E91E63',
      'bce_loss': '#E91E63',
      'adam': '#3F51B5',
      'sgd': '#3F51B5',
      'lstm': '#009688',
      'gru': '#009688',
      'multihead_attention': '#FF5722',
    };
    return colorMap[moduleType] || '#9E9E9E';
  };

  const getNodeIcon = (moduleType: string) => {
    const iconMap: { [key: string]: string } = {
      'input': '📥',
      'output': '📤',
      'linear': '🔗',
      'bilinear': '🔗',
      'conv1d': '📊',
      'conv2d': '🖼️',
      'conv3d': '🎯',
      'maxpool1d': '⬇️',
      'maxpool2d': '⬇️',
      'avgpool2d': '📉',
      'relu': '⚡',
      'sigmoid': '〰️',
      'tanh': '📈',
      'softmax': '🎯',
      'leaky_relu': '⚡',
      'batchnorm1d': '🔄',
      'batchnorm2d': '🔄',
      'layernorm': '🔄',
      'dropout': '❌',
      'dropout2d': '❌',
      'cross_entropy': '🎯',
      'mse_loss': '📏',
      'bce_loss': '⚖️',
      'adam': '🚀',
      'sgd': '📈',
      'lstm': '🔄',
      'gru': '🔄',
      'multihead_attention': '🎯',
    };
    return iconMap[moduleType] || '🔧';
  };

  const shouldShowInputHandle = () => {
    return data.moduleType !== 'input' && data.moduleType !== 'adam' && data.moduleType !== 'sgd';
  };

  const shouldShowOutputHandle = () => {
    return data.moduleType !== 'output' && !data.moduleType.includes('loss') && data.moduleType !== 'adam' && data.moduleType !== 'sgd';
  };

  const formatConfigValue = (key: string, value: any) => {
    if (Array.isArray(value)) {
      return `[${value.join(', ')}]`;
    }
    return value.toString();
  };

  return (
    <div 
      className={`pytorch-node ${selected ? 'selected' : ''}`}
      style={{
        borderColor: getNodeColor(data.moduleType),
        borderWidth: selected ? '2px' : '1px'
      }}
    >
      {shouldShowInputHandle() && (
        <Handle
          type="target"
          position={Position.Top}
          className="node-handle input-handle"
        />
      )}
      
      <div className="node-header" style={{ backgroundColor: getNodeColor(data.moduleType) }}>
        <span className="node-icon">{getNodeIcon(data.moduleType)}</span>
        <span className="node-title">{data.label}</span>
        {Object.keys(data.config).length > 0 && (
          <button 
            className="config-toggle"
            onClick={() => setIsConfigOpen(!isConfigOpen)}
          >
            ⚙️
          </button>
        )}
      </div>

      <div className="node-body">
        {isConfigOpen && Object.keys(data.config).length > 0 && (
          <div className="node-config">
            {Object.entries(data.config).map(([key, value]) => (
              <div key={key} className="config-item">
                <span className="config-key">{key}:</span>
                <span className="config-value">{formatConfigValue(key, value)}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {shouldShowOutputHandle() && (
        <Handle
          type="source"
          position={Position.Bottom}
          className="node-handle output-handle"
        />
      )}
    </div>
  );
});