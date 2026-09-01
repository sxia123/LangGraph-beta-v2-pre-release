document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const sidebar = document.getElementById('sidebar');
  const newChatBtn = document.getElementById('newChatBtn');
  const collapseSidebarBtn = document.getElementById('collapseSidebarBtn');
  const expandSidebarBtn = document.getElementById('expandSidebarBtn');
  const historyList = document.getElementById('historyList');

  // Status & Model Select
  const statusBullet = document.getElementById('statusBullet');
  const statusTitle = document.getElementById('statusTitle');
  const statusDetails = document.getElementById('statusDetails');
  const modelSelectBtn = document.getElementById('modelSelectBtn');
  const currentModelLabel = document.getElementById('currentModelLabel');
  const modelDropdownMenu = document.getElementById('modelDropdownMenu');

  // Pipeline Select & Memory Toggle
  const pipelineSelectBtn = document.getElementById('pipelineSelectBtn');
  const currentPipelineLabel = document.getElementById('currentPipelineLabel');
  const pipelineDropdownMenu = document.getElementById('pipelineDropdownMenu');
  const memoryToggleBtn = document.getElementById('memoryToggleBtn');
  const memoryToggleLabel = document.getElementById('memoryToggleLabel');

  // Chat Viewport & Input
  const chatViewport = document.getElementById('chatViewport');
  const chatThread = document.getElementById('chatThread');
  const emptyState = document.getElementById('emptyState');
  const chatForm = document.getElementById('chatForm');
  const chatTextarea = document.getElementById('chatTextarea');
  const sendBtn = document.getElementById('sendBtn');
  const pauseBtn = document.getElementById('pauseBtn');
  const floatingPauseBar = document.getElementById('floatingPauseBar');
  const floatingPauseBtn = document.getElementById('floatingPauseBtn');
  const attachBtn = document.getElementById('attachBtn');
  const fileInput = document.getElementById('fileInput');
  const imageStagingTray = document.getElementById('imageStagingTray');
  const suggestionCards = document.querySelectorAll('.suggestion-card');

  // Overlays & Modals
  const dragDropOverlay = document.getElementById('dragDropOverlay');
  const lightboxModal = document.getElementById('lightboxModal');
  const lightboxImage = document.getElementById('lightboxImage');
  const lightboxCloseBtn = document.getElementById('lightboxCloseBtn');
  const lightboxBackdrop = document.getElementById('lightboxBackdrop');

  // Memory Modal
  const openMemoryModalBtn = document.getElementById('openMemoryModalBtn');
  const memoryModal = document.getElementById('memoryModal');
  const memoryModalBackdrop = document.getElementById('memoryModalBackdrop');
  const closeMemoryModalBtn = document.getElementById('closeMemoryModalBtn');
  const doneMemoryModalBtn = document.getElementById('doneMemoryModalBtn');
  const memoryBadge = document.getElementById('memoryBadge');
  const memoriesList = document.getElementById('memoriesList');
  const memorySearchInput = document.getElementById('memorySearchInput');
  const showAddMemoryFormBtn = document.getElementById('showAddMemoryFormBtn');
  const addMemoryForm = document.getElementById('addMemoryForm');
  const cancelAddMemoryBtn = document.getElementById('cancelAddMemoryBtn');
  const saveMemoryBtn = document.getElementById('saveMemoryBtn');
  const newMemoryTopic = document.getElementById('newMemoryTopic');
  const newMemoryContent = document.getElementById('newMemoryContent');
  const clearAllMemoriesBtn = document.getElementById('clearAllMemoriesBtn');

  // Tools Modal & Controls
  const openToolsModalBtn = document.getElementById('openToolsModalBtn');
  const topbarToolsBtn = document.getElementById('topbarToolsBtn');
  const inlineToolsBtn = document.getElementById('inlineToolsBtn');
  const inlineToolsBadge = document.getElementById('inlineToolsBadge');
  const emptyToolsSection = document.getElementById('emptyToolsSection');
  const emptyToolsPills = document.getElementById('emptyToolsPills');
  const emptyToolsCount = document.getElementById('emptyToolsCount');
  const emptyToolsExploreBtn = document.getElementById('emptyToolsExploreBtn');
  const toolsSearchInput = document.getElementById('toolsSearchInput');
  const toolsCategoryFilters = document.getElementById('toolsCategoryFilters');
  const catCountAll = document.getElementById('catCountAll');
  const toolsModal = document.getElementById('toolsModal');
  const toolsModalBackdrop = document.getElementById('toolsModalBackdrop');
  const closeToolsModalBtn = document.getElementById('closeToolsModalBtn');
  const doneToolsModalBtn = document.getElementById('doneToolsModalBtn');
  const toolsBadge = document.getElementById('toolsBadge');
  const topbarToolsBadge = document.getElementById('topbarToolsBadge');
  const tabToolsCount = document.getElementById('tabToolsCount');
  const tabBtnCatalog = document.getElementById('tabBtnCatalog');
  const tabBtnTester = document.getElementById('tabBtnTester');
  const tabBtnCheckpoints = document.getElementById('tabBtnCheckpoints');
  const tabPaneCatalog = document.getElementById('tabPaneCatalog');
  const tabPaneTester = document.getElementById('tabPaneTester');
  const tabPaneCheckpoints = document.getElementById('tabPaneCheckpoints');
  const toolsGrid = document.getElementById('toolsGrid');
  const testerToolSelect = document.getElementById('testerToolSelect');
  const testerToolArgs = document.getElementById('testerToolArgs');
  const runToolTesterBtn = document.getElementById('runToolTesterBtn');
  const testerOutputBox = document.getElementById('testerOutputBox');
  const testerStatusBadge = document.getElementById('testerStatusBadge');
  const testerOutputContent = document.getElementById('testerOutputContent');
  const testerCheckpointFooter = document.getElementById('testerCheckpointFooter');
  const checkpointsFilterInput = document.getElementById('checkpointsFilterInput');
  const refreshCheckpointsBtn = document.getElementById('refreshCheckpointsBtn');
  const checkpointsList = document.getElementById('checkpointsList');

  // State
  let activeModel = 'Qwen3.8-27B-oQ6-mtp';
  let activePipeline = 'direct';
  let useMemory = true;
  let stagedImages = []; // Array of Base64 strings
  let stagedFiles = []; // Array of { filename, content, size, type }
  let isGenerating = false;
  let activeAbortController = null;
  let currentRunId = null;
  let registeredTools = [];

  // Configure marked for clean markdown rendering
  if (window.marked) {
    window.marked.setOptions({
      breaks: true,
      gfm: true,
      headerIds: false,
      mangle: false,
    });
  }

  // =========================================================================
  // 1. Check Local LLM Status & Available Models
  // =========================================================================
  // =========================================================================
  // 1. Check Local LLM Status & Available Models
  // =========================================================================
  async function checkServerStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();

      if (data.connection && data.connection.ok) {
        statusBullet.className = 'status-bullet online';
        statusTitle.textContent = `${data.provider.toUpperCase()} Online`;
        statusDetails.textContent = `${data.model_name}`;
        if (data.model_name && (!activeModel || activeModel === 'Qwen3.8-27B-oQ6-mtp')) {
          activeModel = data.model_name;
          currentModelLabel.textContent = formatModelLabel(data.model_name);
        }
      } else {
        statusBullet.className = 'status-bullet';
        statusTitle.textContent = 'oMLX Offline (Mock Ready)';
        statusDetails.textContent = `${data.base_url}`;
      }

      if (data.tools_count !== undefined) {
        if (toolsBadge) toolsBadge.textContent = data.tools_count;
        if (topbarToolsBadge) topbarToolsBadge.textContent = data.tools_count;
      }

      // Populate models dynamically from oMLX / LLM server
      const detectedModels = data.models || (data.connection && data.connection.models) || [];
      renderModelDropdown(detectedModels);
    } catch {
      statusBullet.className = 'status-bullet';
      statusTitle.textContent = 'Server Disconnected';
      statusDetails.textContent = 'Port 8080 unavailable';
    }
  }

  function formatModelLabel(modelId) {
    if (!modelId) return 'Select Model';
    if (modelId.includes('Qwen3.8-27B')) return 'Qwen 3.8 27B';
    if (modelId.includes('Qwen2.5-VL-72B')) return 'Qwen 2.5 VL 72B';
    if (modelId.includes('Qwen2.5-VL-7B')) return 'Qwen 2.5 VL 7B';
    if (modelId.includes('Qwen2.5-Coder-32B')) return 'Qwen 2.5 Coder 32B';
    if (modelId.toLowerCase().includes('deepseek-r1')) return 'DeepSeek R1';
    if (modelId.toLowerCase().includes('llama-3.2') || modelId.toLowerCase().includes('llama3.2')) return 'Llama 3.2';

    // Format clean readable label from path or repo ID
    const clean = modelId.includes('/') ? modelId.split('/').pop() : modelId;
    return clean;
  }

  function renderModelDropdown(modelsList) {
    if (!modelDropdownMenu) return;
    let list = Array.isArray(modelsList) ? [...modelsList] : [];
    if (list.length === 0) {
      list = [
        'Qwen3.8-27B-oQ6-mtp',
        'Qwen2.5-VL-72B-Instruct',
        'Qwen2.5-VL-7B-Instruct',
        'Qwen2.5-Coder-32B-Instruct',
      ];
    }

    if (activeModel && !list.includes(activeModel)) {
      list.unshift(activeModel);
    }

    modelDropdownMenu.innerHTML = '';
    const header = document.createElement('div');
    header.className = 'dropdown-header';
    header.textContent = `oMLX Models (${list.length})`;
    modelDropdownMenu.appendChild(header);

    list.forEach((mId) => {
      const btn = document.createElement('button');
      btn.className = `dropdown-item ${mId === activeModel ? 'active' : ''}`;
      btn.setAttribute('data-model', mId);

      const title = formatModelLabel(mId);
      const isDefault = mId === activeModel;

      btn.innerHTML = `
        <div class="item-title">${escapeHtml(title)} ${isDefault ? '(Active)' : ''}</div>
        <div class="item-desc" style="font-family: var(--font-mono); font-size: 10.5px; opacity: 0.75;">${escapeHtml(mId)}</div>
      `;

      btn.addEventListener('click', async () => {
        modelDropdownMenu.querySelectorAll('.dropdown-item').forEach((i) => i.classList.remove('active'));
        btn.classList.add('active');
        activeModel = mId;
        currentModelLabel.textContent = title;
        modelDropdownMenu.classList.remove('show');

        try {
          await fetch('/api/models/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: mId }),
          });
        } catch {
          // Ignore
        }
      });

      modelDropdownMenu.appendChild(btn);
    });

    // Custom model input row
    const customRow = document.createElement('div');
    customRow.className = 'dropdown-custom-input-row';
    customRow.innerHTML = `
      <input type="text" class="dropdown-custom-input" id="customModelInput" placeholder="Custom model ID...">
      <button type="button" class="dropdown-custom-btn" id="applyCustomModelBtn">Set</button>
    `;

    customRow.addEventListener('click', (e) => {
      e.stopPropagation();
    });

    const customInput = customRow.querySelector('#customModelInput');
    const applyBtn = customRow.querySelector('#applyCustomModelBtn');

    const applyCustomModel = async () => {
      const val = customInput.value.trim();
      if (!val) return;
      activeModel = val;
      currentModelLabel.textContent = formatModelLabel(val);
      renderModelDropdown([...list, val]);
      modelDropdownMenu.classList.remove('show');
      try {
        await fetch('/api/models/select', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_name: val }),
        });
      } catch {
        // Ignore
      }
    };

    applyBtn.addEventListener('click', applyCustomModel);
    customInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        applyCustomModel();
      }
    });

    modelDropdownMenu.appendChild(customRow);
  }

  checkServerStatus();
  setInterval(checkServerStatus, 20000);

  // =========================================================================
  // 2. Sidebar & Navigation Controls
  // =========================================================================
  collapseSidebarBtn.addEventListener('click', () => {
    sidebar.classList.add('collapsed');
  });

  expandSidebarBtn.addEventListener('click', () => {
    sidebar.classList.remove('collapsed');
  });

  newChatBtn.addEventListener('click', () => {
    startNewChat();
  });

  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      startNewChat();
    }
  });

  function startNewChat() {
    currentRunId = null;
    chatThread.innerHTML = '';
    chatThread.appendChild(emptyState);
    emptyState.style.display = 'flex';
    clearStagedAttachments();
    chatTextarea.value = '';
    autoResizeTextarea();
    updateSendButtonState();
    chatTextarea.focus();
  }

  // =========================================================================
  // 3. Dropdowns (Model & Pipeline) & Memory Toggle
  // =========================================================================
  modelSelectBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    pipelineDropdownMenu.classList.remove('show');
    modelDropdownMenu.classList.toggle('show');
  });

  pipelineSelectBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    modelDropdownMenu.classList.remove('show');
    pipelineDropdownMenu.classList.toggle('show');
  });

  document.addEventListener('click', () => {
    modelDropdownMenu.classList.remove('show');
    pipelineDropdownMenu.classList.remove('show');
  });

  pipelineDropdownMenu.querySelectorAll('.dropdown-item').forEach((item) => {
    item.addEventListener('click', () => {
      pipelineDropdownMenu.querySelectorAll('.dropdown-item').forEach((i) => i.classList.remove('active'));
      item.classList.add('active');
      activePipeline = item.getAttribute('data-pipeline');
      currentPipelineLabel.textContent = item.querySelector('.item-title').textContent;
      pipelineDropdownMenu.classList.remove('show');
    });
  });

  memoryToggleBtn.addEventListener('click', () => {
    useMemory = !useMemory;
    if (useMemory) {
      memoryToggleBtn.classList.add('active');
      memoryToggleLabel.textContent = 'Memory: On';
    } else {
      memoryToggleBtn.classList.remove('active');
      memoryToggleLabel.textContent = 'Memory: Off';
    }
  });

  // Suggestion prompt chips
  suggestionCards.forEach((card) => {
    card.addEventListener('click', () => {
      const prompt = card.getAttribute('data-prompt');
      chatTextarea.value = prompt;
      autoResizeTextarea();
      updateSendButtonState();
      chatTextarea.focus();
    });
  });

  // =========================================================================
  // 4. Multimodal File & Image Intake (File Input, Drag & Drop, Clipboard)
  // =========================================================================
  attachBtn.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files.length > 0) {
      Array.from(fileInput.files).forEach(handleFileInput);
      fileInput.value = '';
    }
  });

  function getFileIcon(filename) {
    if (!filename) return '📁';
    const ext = filename.split('.').pop().toLowerCase();
    if (['py'].includes(ext)) return '🐍';
    if (['js', 'ts', 'jsx', 'tsx'].includes(ext)) return '📜';
    if (['json'].includes(ext)) return '📋';
    if (['md', 'txt', 'rst', 'log'].includes(ext)) return '📝';
    if (['pdf', 'docx', 'doc'].includes(ext)) return '📄';
    if (['csv', 'tsv', 'xlsx', 'xls'].includes(ext)) return '📊';
    if (['html', 'css', 'scss', 'xml'].includes(ext)) return '🌐';
    if (['yaml', 'yml', 'toml', 'env', 'ini', 'conf'].includes(ext)) return '⚙️';
    if (['sh', 'bash', 'zsh', 'bat', 'ps1'].includes(ext)) return '💻';
    if (['sql', 'db', 'sqlite'].includes(ext)) return '🗄️';
    if (['png', 'jpg', 'jpeg', 'webp', 'gif', 'svg', 'bmp'].includes(ext)) return '🖼️';
    return '📄';
  }

  function formatFileSize(bytes) {
    if (!bytes || bytes <= 0) return '0 B';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function handleFileInput(file) {
    if (!file) return;

    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        stagedImages.push(e.target.result);
        renderStagedAttachments();
      };
      reader.readAsDataURL(file);
    } else {
      const isBinaryDoc = /\.(xlsx|xls|xlsm|docx|pptx|pdf|zip|tar|gz|bin)$/i.test(file.name) ||
                          file.type.includes('spreadsheet') ||
                          file.type.includes('excel') ||
                          file.type.includes('officedocument');

      const reader = new FileReader();
      reader.onload = (e) => {
        stagedFiles.push({
          filename: file.name,
          content: e.target.result,
          size: file.size,
          type: file.type || (isBinaryDoc ? 'application/octet-stream' : 'text/plain'),
        });
        renderStagedAttachments();
      };
      reader.onerror = () => {
        const binReader = new FileReader();
        binReader.onload = (be) => {
          stagedFiles.push({
            filename: file.name,
            content: be.target.result,
            size: file.size,
            type: file.type || 'application/octet-stream',
          });
          renderStagedAttachments();
        };
        binReader.readAsDataURL(file);
      };

      if (isBinaryDoc) {
        reader.readAsDataURL(file);
      } else {
        reader.readAsText(file);
      }
    }
  }

  function renderStagedAttachments() {
    imageStagingTray.innerHTML = '';

    // Render Staged Images
    stagedImages.forEach((imgData, index) => {
      const chip = document.createElement('div');
      chip.className = 'stage-chip';
      chip.innerHTML = `
        <img src="${imgData}" alt="Attachment ${index + 1}">
        <button type="button" class="remove-chip-btn" title="Remove image">×</button>
      `;
      chip.querySelector('.remove-chip-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        stagedImages.splice(index, 1);
        renderStagedAttachments();
      });
      imageStagingTray.appendChild(chip);
    });

    // Render Staged Files
    stagedFiles.forEach((fileObj, index) => {
      const chip = document.createElement('div');
      chip.className = 'file-stage-chip';
      const icon = getFileIcon(fileObj.filename);
      const sizeStr = formatFileSize(fileObj.size);

      chip.innerHTML = `
        <span class="file-icon">${icon}</span>
        <div class="file-info">
          <span class="file-name" title="${escapeHtml(fileObj.filename)}">${escapeHtml(fileObj.filename)}</span>
          <span class="file-size">${escapeHtml(sizeStr)}</span>
        </div>
        <button type="button" class="remove-chip-btn" title="Remove file">×</button>
      `;

      chip.querySelector('.remove-chip-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        stagedFiles.splice(index, 1);
        renderStagedAttachments();
      });

      imageStagingTray.appendChild(chip);
    });

    updateSendButtonState();
  }

  function clearStagedAttachments() {
    stagedImages = [];
    stagedFiles = [];
    imageStagingTray.innerHTML = '';
    updateSendButtonState();
  }

  // Drag and Drop
  let dragCounter = 0;
  window.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragCounter++;
    if (e.dataTransfer && e.dataTransfer.types && e.dataTransfer.types.includes('Files')) {
      dragDropOverlay.classList.add('active');
    }
  });

  window.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter <= 0) {
      dragCounter = 0;
      dragDropOverlay.classList.remove('active');
    }
  });

  window.addEventListener('dragover', (e) => {
    e.preventDefault();
  });

  window.addEventListener('drop', (e) => {
    e.preventDefault();
    dragCounter = 0;
    dragDropOverlay.classList.remove('active');

    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      Array.from(e.dataTransfer.files).forEach(handleFileInput);
    }
  });

  // Clipboard Paste (Ctrl+V screenshot / file intake)
  window.addEventListener('paste', (e) => {
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;

    for (let i = 0; i < items.length; i++) {
      if (items[i].kind === 'file') {
        const blob = items[i].getAsFile();
        if (blob) {
          handleFileInput(blob);
        }
      }
    }
  });

  // Lightbox Modal
  function openLightbox(src) {
    lightboxImage.src = src;
    lightboxModal.classList.add('active');
  }

  function closeLightbox() {
    lightboxModal.classList.remove('active');
    lightboxImage.src = '';
  }

  lightboxCloseBtn.addEventListener('click', closeLightbox);
  lightboxBackdrop.addEventListener('click', closeLightbox);
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lightboxModal.classList.contains('active')) {
      closeLightbox();
    }
  });

  // =========================================================================
  // 5. Input Auto-Resize & Submit Handling
  // =========================================================================
  function autoResizeTextarea() {
    chatTextarea.style.height = 'auto';
    chatTextarea.style.height = Math.min(chatTextarea.scrollHeight, 180) + 'px';
  }

  function updateSendButtonState() {
    const hasText = chatTextarea.value.trim().length > 0;
    const hasImages = stagedImages.length > 0;
    const hasFiles = stagedFiles.length > 0;

    if (isGenerating) {
      sendBtn.style.display = 'none';
      if (pauseBtn) pauseBtn.style.display = 'flex';
      if (floatingPauseBar) floatingPauseBar.classList.add('show');
    } else {
      if (pauseBtn) pauseBtn.style.display = 'none';
      if (floatingPauseBar) floatingPauseBar.classList.remove('show');
      sendBtn.style.display = 'flex';
      sendBtn.disabled = !(hasText || hasImages || hasFiles);
    }
  }

  async function pauseGeneration() {
    if (!isGenerating) return;
    if (activeAbortController) {
      activeAbortController.abort();
      activeAbortController = null;
    }
    try {
      await fetch('/api/chat/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: currentRunId }),
      });
    } catch {
      // Ignore cancellation fetch error
    }
  }

  if (pauseBtn) {
    pauseBtn.addEventListener('click', pauseGeneration);
  }

  if (floatingPauseBtn) {
    floatingPauseBtn.addEventListener('click', pauseGeneration);
  }

  chatTextarea.addEventListener('input', () => {
    autoResizeTextarea();
    updateSendButtonState();
  });

  chatTextarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!sendBtn.disabled && !isGenerating) {
        chatForm.dispatchEvent(new Event('submit'));
      }
    }
  });

  function getAgentBadgeClass(agentName) {
    const name = String(agentName || '').toLowerCase();
    if (name.includes('tool')) return 'tool-badge';
    if (name.includes('supervisor')) return 'badge-purple';
    if (name.includes('research')) return 'badge-cyan';
    if (name.includes('coder') || name.includes('dev')) return 'badge-emerald';
    if (name.includes('critic') || name.includes('audit') || name.includes('review')) return 'badge-amber';
    if (name.includes('specialist') || name.includes('solution')) return 'badge-indigo';
    if (name.includes('class') || name.includes('triage') || name.includes('intake')) return 'badge-cyan';
    return 'badge-blue';
  }

  function createThoughtCard(agent, thoughtText, timestamp) {
    const card = document.createElement('div');
    card.className = 'thought-step-card';

    const badgeClass = getAgentBadgeClass(agent);
    const timeStr = timestamp || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    let renderedContent = '';
    if (window.marked && typeof thoughtText === 'string') {
      try {
        renderedContent = window.marked.parse(thoughtText);
      } catch {
        renderedContent = `<p>${escapeHtml(thoughtText)}</p>`;
      }
    } else {
      renderedContent = `<p>${escapeHtml(typeof thoughtText === 'object' ? JSON.stringify(thoughtText, null, 2) : String(thoughtText))}</p>`;
    }

    card.innerHTML = `
      <div class="thought-step-header">
        <span class="thought-agent-badge ${badgeClass}">${escapeHtml(agent || 'Reasoning')}</span>
        <span class="thought-step-time">${escapeHtml(timeStr)}</span>
      </div>
      <div class="thought-step-text">${renderedContent}</div>
    `;
    return card;
  }

  // Form Submit & SSE Stream
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const promptText = chatTextarea.value.trim();
    if (!promptText && stagedImages.length === 0 && stagedFiles.length === 0) return;
    if (isGenerating) return;

    const currentAttachments = [...stagedImages];
    const currentFiles = [...stagedFiles];

    // Hide welcome state
    emptyState.style.display = 'none';

    // Append User Message to Thread
    appendUserMessage(promptText, currentAttachments, currentFiles);

    // Clear input & staging
    chatTextarea.value = '';
    clearStagedAttachments();
    autoResizeTextarea();
    activeAbortController = new AbortController();
    isGenerating = true;
    updateSendButtonState();

    // Append Assistant Placeholder Container
    const assistantRow = appendAssistantPlaceholder();
    const thoughtBox = assistantRow.querySelector('.thought-accordion');
    const thoughtContent = assistantRow.querySelector('.thought-content');
    const thoughtCountLabel = assistantRow.querySelector('.thought-count-label');
    const thoughtTimeBadge = assistantRow.querySelector('.thought-time-badge');
    const markdownBody = assistantRow.querySelector('.markdown-body');
    const executingIndicator = assistantRow.querySelector('.executing-indicator');

    let thoughtEntries = [];
    assistantRow._rawThoughts = thoughtEntries;
    let streamedSteps = [];
    let finalAnswerText = '';

    const startTime = Date.now();
    const timerInterval = setInterval(() => {
      if (!isGenerating) return;
      const elapsedSec = ((Date.now() - startTime) / 1000).toFixed(1);
      if (thoughtTimeBadge) {
        thoughtTimeBadge.textContent = `${elapsedSec}s`;
        if (thoughtBox && thoughtBox.style.display !== 'none') {
          thoughtTimeBadge.style.display = 'inline-block';
        }
      }
    }, 100);

    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: activeAbortController.signal,
        body: JSON.stringify({
          prompt: promptText,
          images: currentAttachments.length > 0 ? currentAttachments : null,
          files: currentFiles.length > 0 ? currentFiles : null,
          pipeline: activePipeline,
          model_name: activeModel,
          use_memory: useMemory,
        }),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep partial line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;

          let event;
          try {
            event = JSON.parse(jsonStr);
          } catch {
            continue;
          }

          if (event.type === 'step') {
            const stepData = event.data;
            streamedSteps.push(stepData);

            // Collect and render thoughts
            if (stepData.thoughts && stepData.thoughts.length > 0) {
              stepData.thoughts.forEach((t) => {
                const thoughtText = typeof t === 'string' ? t : t.thought || JSON.stringify(t);
                const agentName = (typeof t === 'object' && t.agent) ? t.agent : (stepData.node || 'Reasoning');
                const timeVal = (typeof t === 'object' && t.timestamp) ? t.timestamp : '';

                thoughtEntries.push({ agent: agentName, thought: thoughtText, timestamp: timeVal });

                const card = createThoughtCard(agentName, thoughtText, timeVal);
                thoughtContent.appendChild(card);
              });
            }

            // Update thought accordion live
            if (thoughtEntries.length > 0) {
              thoughtBox.style.display = 'block';
              thoughtBox.classList.add('thinking');
              thoughtBox.classList.add('open');
              if (thoughtTimeBadge) thoughtTimeBadge.style.display = 'inline-block';
              thoughtCountLabel.textContent = `Thinking (${thoughtEntries.length} step${thoughtEntries.length > 1 ? 's' : ''})...`;
            }

            // If there's partial final response or message
            if (stepData.final_response) {
              finalAnswerText = stepData.final_response;
              renderMarkdown(markdownBody, finalAnswerText);
            }
          } else if (event.type === 'complete') {
            finalAnswerText = event.final_response || finalAnswerText || 'Completed.';
            renderMarkdown(markdownBody, finalAnswerText);
            currentRunId = event.run_id;
            refreshRunsHistory();
            refreshMemoryMetrics();
          } else if (event.type === 'error') {
            throw new Error(event.detail || 'Execution error occurred');
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        if (!finalAnswerText) {
          finalAnswerText = '[Generation paused by user]';
        }
        renderMarkdown(markdownBody, finalAnswerText);
      } else {
        markdownBody.innerHTML = `<div style="color: var(--status-offline); padding: 8px 0;">Error: ${escapeHtml(err.message)}</div>`;
      }
    } finally {
      activeAbortController = null;
      clearInterval(timerInterval);
      const totalElapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      if (thoughtBox) {
        thoughtBox.classList.remove('thinking');
        if (thoughtEntries.length > 0) {
          thoughtCountLabel.textContent = `Thought for ${totalElapsed}s (${thoughtEntries.length} step${thoughtEntries.length > 1 ? 's' : ''})`;
          if (thoughtTimeBadge) {
            thoughtTimeBadge.textContent = `${totalElapsed}s`;
            thoughtTimeBadge.style.display = 'inline-block';
          }
        }
      }
      executingIndicator.style.display = 'none';
      isGenerating = false;
      updateSendButtonState();
      scrollToBottom();
    }
  });

  function appendUserMessage(text, images, files) {
    const row = document.createElement('div');
    row.className = 'message-row user';

    let imagesHtml = '';
    if (images && images.length > 0) {
      imagesHtml = `<div class="attached-images-grid">`;
      images.forEach((imgSrc) => {
        imagesHtml += `<img src="${imgSrc}" class="chat-image-thumbnail" alt="Attached image">`;
      });
      imagesHtml += `</div>`;
    }

    let filesHtml = '';
    if (files && files.length > 0) {
      filesHtml = `<div class="attached-files-list">`;
      files.forEach((f) => {
        const fn = typeof f === 'string' ? f : (f.filename || 'File');
        const icon = getFileIcon(fn);
        const sizeStr = (typeof f === 'object' && f.size) ? formatFileSize(f.size) : '';
        filesHtml += `
          <div class="user-file-chip">
            <span class="file-chip-icon">${icon}</span>
            <span class="file-chip-name">${escapeHtml(fn)}</span>
            ${sizeStr ? `<span class="file-chip-size">(${escapeHtml(sizeStr)})</span>` : ''}
          </div>
        `;
      });
      filesHtml += `</div>`;
    }

    row.innerHTML = `
      <div class="user-message-bubble">
        ${imagesHtml}
        ${filesHtml}
        ${text ? `<div class="user-text">${escapeHtml(text)}</div>` : ''}
      </div>
    `;

    // Add lightbox listeners to thumbnail images
    row.querySelectorAll('.chat-image-thumbnail').forEach((img) => {
      img.addEventListener('click', () => openLightbox(img.src));
    });

    chatThread.appendChild(row);
    scrollToBottom();
  }

  function appendAssistantPlaceholder() {
    const row = document.createElement('div');
    row.className = 'message-row assistant';
    row.innerHTML = `
      <div class="assistant-avatar">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
        </svg>
      </div>
      <div class="assistant-body">
        <div class="thought-accordion" style="display: none;">
          <div class="thought-header">
            <div class="thought-header-left">
              <div class="thought-icon-wrapper">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M12 2a4 4 0 0 1 4 4c0 1.1-.5 2.1-1.2 2.8L12 11l-2.8-2.2A4 4 0 0 1 12 2z"/>
                  <path d="M4.2 10.2a8 8 0 1 0 15.6 0"/>
                  <path d="M12 11v11"/>
                </svg>
              </div>
              <span class="thought-count-label">Thinking...</span>
            </div>
            <div class="thought-header-right">
              <span class="thought-time-badge" style="display: none;">0.0s</span>
              <button type="button" class="copy-thought-btn" title="Copy reasoning">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
                <span>Copy</span>
              </button>
              <svg class="thought-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
            </div>
          </div>
          <div class="thought-content"></div>
        </div>

        <div class="markdown-body"></div>

        <div class="executing-indicator">
          <span class="pulse-dot"></span>
          <span>Thinking...</span>
        </div>
      </div>
    `;

    const accordion = row.querySelector('.thought-accordion');
    const header = row.querySelector('.thought-header');
    const copyBtn = row.querySelector('.copy-thought-btn');

    header.addEventListener('click', (e) => {
      if (e.target.closest('.copy-thought-btn')) return;
      accordion.classList.toggle('open');
    });

    copyBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const rawThoughts = row._rawThoughts || [];
      const textToCopy = rawThoughts.length > 0
        ? rawThoughts.map((t) => `[${t.agent || 'Reasoning'}]: ${t.thought}`).join('\n\n')
        : (row.querySelector('.thought-content')?.innerText || '');

      navigator.clipboard.writeText(textToCopy).then(() => {
        const span = copyBtn.querySelector('span');
        if (span) {
          span.textContent = 'Copied!';
          setTimeout(() => { span.textContent = 'Copy'; }, 2000);
        }
      }).catch((err) => console.error('Copy failed', err));
    });

    chatThread.appendChild(row);
    scrollToBottom();
    return row;
  }

  if (window.mermaid) {
    window.mermaid.initialize({
      startOnLoad: false,
      theme: 'dark',
      themeVariables: {
        darkMode: true,
        background: '#212121',
        primaryColor: '#2f2f2f',
        primaryTextColor: '#ececec',
        primaryBorderColor: '#383838',
        lineColor: '#b4b4b4',
        secondaryColor: '#262626',
        tertiaryColor: '#171717',
      },
    });
  }

  function renderMarkdown(container, text) {
    if (window.marked) {
      container.innerHTML = window.marked.parse(text);

      // Render Mermaid flowchart/chart diagrams if present
      if (window.mermaid) {
        container.querySelectorAll('pre code.language-mermaid').forEach(async (el) => {
          const pre = el.parentElement;
          const mermaidCode = el.innerText;
          const id = 'mermaid_' + Math.random().toString(36).substring(2, 9);
          try {
            const { svg } = await window.mermaid.render(id, mermaidCode);
            const div = document.createElement('div');
            div.className = 'mermaid-chart-container';
            div.style.padding = '12px';
            div.style.textAlign = 'center';
            div.style.overflowX = 'auto';
            div.innerHTML = svg;
            if (pre.parentNode) {
              pre.parentNode.replaceChild(div, pre);
            }
          } catch {
            // Keep original pre code block on render error
          }
        });
      }

      // Enhance code blocks with copy buttons
      container.querySelectorAll('pre').forEach((pre) => {
        if (!pre.querySelector('.code-header')) {
          const codeEl = pre.querySelector('code');
          const langMatch = codeEl ? codeEl.className.match(/language-(\w+)/) : null;
          const lang = langMatch ? langMatch[1] : 'code';

          if (lang === 'mermaid') return;

          const header = document.createElement('div');
          header.className = 'code-header';
          header.innerHTML = `
            <span>${lang}</span>
            <button type="button">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
              <span>Copy code</span>
            </button>
          `;

          const copyBtn = header.querySelector('button');
          copyBtn.addEventListener('click', () => {
            const rawCode = codeEl ? codeEl.innerText : pre.innerText;
            navigator.clipboard.writeText(rawCode).then(() => {
              copyBtn.querySelector('span').textContent = 'Copied!';
              setTimeout(() => {
                copyBtn.querySelector('span').textContent = 'Copy code';
              }, 2000);
            });
          });

          pre.insertBefore(header, pre.firstChild);
        }
      });
    } else {
      container.textContent = text;
    }
  }

  function scrollToBottom() {
    chatViewport.scrollTop = chatViewport.scrollHeight;
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // =========================================================================
  // 6. Persistent Memory Management Modal
  // =========================================================================
  async function refreshMemoryMetrics() {
    try {
      const res = await fetch('/api/memory-summary');
      const data = await res.json();
      memoryBadge.textContent = data.total_memories || 0;
    } catch {
      memoryBadge.textContent = '0';
    }
  }

  async function loadMemories(searchQuery = '') {
    memoriesList.innerHTML = '<div class="loading-spinner">Loading stored memories...</div>';
    try {
      const url = searchQuery
        ? `/api/memories?q=${encodeURIComponent(searchQuery)}`
        : '/api/memories';
      const res = await fetch(url);
      const data = await res.json();
      const memories = data.memories || [];

      if (memories.length === 0) {
        memoriesList.innerHTML = `<div class="history-empty">${searchQuery ? 'No matching memories found' : 'No memories saved yet. Memories are automatically recorded during chats.'}</div>`;
        return;
      }

      memoriesList.innerHTML = '';
      memories.forEach((m) => {
        const card = document.createElement('div');
        card.className = 'memory-card';

        let resText = '';
        if (m.result) {
          resText = typeof m.result === 'object' ? JSON.stringify(m.result) : String(m.result);
        } else if (m.input) {
          resText = typeof m.input === 'object' ? JSON.stringify(m.input) : String(m.input);
        }

        card.innerHTML = `
          <div class="memory-card-body">
            <span class="memory-tag">${escapeHtml(m.event || 'memory')}</span>
            <div class="memory-text">${escapeHtml(resText.slice(0, 300))}</div>
            <div class="memory-time">${escapeHtml(m.timestamp || '')}</div>
          </div>
          <button class="delete-memory-btn" data-id="${m.id}" title="Delete memory">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="3 6 5 6 21 6"></polyline>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
            </svg>
          </button>
        `;

        card.querySelector('.delete-memory-btn').addEventListener('click', async () => {
          await deleteMemory(m.id);
        });

        memoriesList.appendChild(card);
      });
    } catch {
      memoriesList.innerHTML = '<div class="history-empty">Could not load memories</div>';
    }
  }

  async function deleteMemory(memoryId) {
    try {
      const res = await fetch(`/api/memories/${memoryId}`, { method: 'DELETE' });
      if (res.ok) {
        loadMemories(memorySearchInput.value.trim());
        refreshMemoryMetrics();
      }
    } catch (err) {
      console.error('Delete error', err);
    }
  }

  openMemoryModalBtn.addEventListener('click', () => {
    memoryModal.classList.add('active');
    loadMemories();
  });

  function closeMemoryModal() {
    memoryModal.classList.remove('active');
    addMemoryForm.style.display = 'none';
    newMemoryTopic.value = '';
    newMemoryContent.value = '';
  }

  closeMemoryModalBtn.addEventListener('click', closeMemoryModal);
  doneMemoryModalBtn.addEventListener('click', closeMemoryModal);
  memoryModalBackdrop.addEventListener('click', closeMemoryModal);

  showAddMemoryFormBtn.addEventListener('click', () => {
    addMemoryForm.style.display = addMemoryForm.style.display === 'none' ? 'flex' : 'none';
    if (addMemoryForm.style.display === 'flex') {
      newMemoryTopic.focus();
    }
  });

  cancelAddMemoryBtn.addEventListener('click', () => {
    addMemoryForm.style.display = 'none';
    newMemoryTopic.value = '';
    newMemoryContent.value = '';
  });

  saveMemoryBtn.addEventListener('click', async () => {
    const topic = newMemoryTopic.value.trim() || 'user_fact';
    const content = newMemoryContent.value.trim();
    if (!content) return;

    try {
      const res = await fetch('/api/memories', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          event: topic,
          result: content,
          metadata: { source: 'manual_entry' },
        }),
      });

      if (res.ok) {
        addMemoryForm.style.display = 'none';
        newMemoryTopic.value = '';
        newMemoryContent.value = '';
        loadMemories(memorySearchInput.value.trim());
        refreshMemoryMetrics();
      }
    } catch (err) {
      console.error('Save memory error', err);
    }
  });

  let searchTimeout = null;
  memorySearchInput.addEventListener('input', () => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      loadMemories(memorySearchInput.value.trim());
    }, 300);
  });

  clearAllMemoriesBtn.addEventListener('click', async () => {
    if (confirm('Are you sure you want to clear all persistent memories?')) {
      try {
        const res = await fetch('/api/memories', { method: 'DELETE' });
        if (res.ok) {
          loadMemories();
          refreshMemoryMetrics();
        }
      } catch (err) {
        console.error('Clear all memories error', err);
      }
    }
  });

  // =========================================================================
  // 7. Recent Runs / Session History
  // =========================================================================
  async function refreshRunsHistory() {
    try {
      const res = await fetch('/api/runs?limit=30');
      const data = await res.json();
      const runs = data.runs || [];

      if (runs.length === 0) {
        historyList.innerHTML = '<div class="history-empty">No previous chats</div>';
        return;
      }

      historyList.innerHTML = '';
      runs.forEach((r) => {
        const item = document.createElement('div');
        item.className = 'history-item';
        if (currentRunId === r.run_id) item.classList.add('active');

        const titleText = (r.input || 'Untitled Task').trim().slice(0, 32);

        item.innerHTML = `
          <span class="history-text">${escapeHtml(titleText)}</span>
          <button type="button" class="delete-run-btn" data-run-id="${r.run_id}" title="Delete chat">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="3 6 5 6 21 6"></polyline>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
            </svg>
          </button>
        `;

        item.querySelector('.history-text').addEventListener('click', () => {
          loadRunDetails(r.run_id);
        });

        item.querySelector('.delete-run-btn').addEventListener('click', async (e) => {
          e.stopPropagation();
          await deleteRun(r.run_id);
        });

        historyList.appendChild(item);
      });
    } catch {
      historyList.innerHTML = '<div class="history-empty">History unavailable</div>';
    }
  }

  async function loadRunDetails(runId) {
    try {
      const res = await fetch(`/api/runs/${runId}`);
      if (!res.ok) return;
      const data = await res.json();
      const run = data.run;
      const memories = data.memories || [];

      currentRunId = runId;
      chatThread.innerHTML = '';
      emptyState.style.display = 'none';

      // Rebuild User Message
      appendUserMessage(run.input || '', []);

      // Rebuild Assistant Message
      const assistantRow = appendAssistantPlaceholder();
      assistantRow.querySelector('.executing-indicator').style.display = 'none';
      const thoughtBox = assistantRow.querySelector('.thought-accordion');
      const thoughtContent = assistantRow.querySelector('.thought-content');
      const thoughtCountLabel = assistantRow.querySelector('.thought-count-label');
      const markdownBody = assistantRow.querySelector('.markdown-body');

      if (memories.length > 0) {
        thoughtBox.style.display = 'block';
        thoughtBox.classList.remove('thinking');
        thoughtContent.innerHTML = '';
        const rawList = [];
        memories.forEach((m) => {
          const resVal = m.result || m.input || '';
          const txt = typeof resVal === 'object' ? JSON.stringify(resVal, null, 2) : String(resVal);
          const agentName = m.event || 'Recorded Memory';
          rawList.push({ agent: agentName, thought: txt, timestamp: m.timestamp });
          const card = createThoughtCard(agentName, txt, m.timestamp);
          thoughtContent.appendChild(card);
        });
        assistantRow._rawThoughts = rawList;
        thoughtCountLabel.textContent = `Recorded thoughts (${memories.length} step${memories.length > 1 ? 's' : ''})`;
      }

      renderMarkdown(markdownBody, run.final_answer || 'No response recorded.');
      refreshRunsHistory();
    } catch {
      // Ignore load error
    }
  }

  async function deleteRun(runId) {
    try {
      const res = await fetch(`/api/runs/${runId}`, { method: 'DELETE' });
      if (res.ok) {
        if (currentRunId === runId) {
          startNewChat();
        }
        refreshRunsHistory();
        refreshMemoryMetrics();
      }
    } catch {
      // Ignore delete error
    }
  }

  // =========================================================================
  // 8. Tools Registry, Diagnostics & Interactive Tester
  // =========================================================================
  let currentToolsCategory = 'all';
  let currentToolsSearch = '';

  if (openToolsModalBtn) {
    openToolsModalBtn.addEventListener('click', () => {
      openToolsModal();
    });
  }

  if (topbarToolsBtn) {
    topbarToolsBtn.addEventListener('click', () => {
      openToolsModal();
    });
  }

  if (inlineToolsBtn) {
    inlineToolsBtn.addEventListener('click', () => {
      openToolsModal();
    });
  }

  if (emptyToolsExploreBtn) {
    emptyToolsExploreBtn.addEventListener('click', () => {
      openToolsModal();
    });
  }

  if (closeToolsModalBtn) {
    closeToolsModalBtn.addEventListener('click', () => {
      closeToolsModal();
    });
  }

  if (doneToolsModalBtn) {
    doneToolsModalBtn.addEventListener('click', () => {
      closeToolsModal();
    });
  }

  if (toolsModalBackdrop) {
    toolsModalBackdrop.addEventListener('click', () => {
      closeToolsModal();
    });
  }

  function openToolsModal() {
    if (toolsModal) {
      toolsModal.classList.add('active');
      toolsModal.classList.add('open');
    }
    fetchTools();
  }

  function closeToolsModal() {
    if (toolsModal) {
      toolsModal.classList.remove('active');
      toolsModal.classList.remove('open');
    }
  }

  function getToolCategory(toolName) {
    if (['web_search', 'wikipedia', 'arxiv', 'web_scrape'].includes(toolName)) return 'search';
    if (['python_repl', 'math_eval'].includes(toolName)) return 'code';
    if (['doc_convert', 'docling_parse'].includes(toolName)) return 'docs';
    if (['file_write', 'file_edit', 'vector_memory'].includes(toolName)) return 'files';
    if (toolName.startsWith('github_')) return 'git';
    return 'other';
  }

  function getToolCategoryLabel(cat) {
    const labels = {
      search: 'Search & Web',
      code: 'Code & Execution',
      docs: 'Doc Parsing',
      files: 'Files & Checkpoint',
      git: 'Git Control',
      other: 'Utility',
    };
    return labels[cat] || cat;
  }

  function getToolIcon(toolName) {
    if (toolName === 'web_search') return '🔍';
    if (toolName === 'wikipedia') return '📖';
    if (toolName === 'arxiv') return '📑';
    if (toolName === 'python_repl') return '🐍';
    if (toolName === 'math_eval') return '➗';
    if (toolName === 'web_scrape') return '🌐';
    if (toolName === 'doc_convert') return '📄';
    if (toolName === 'docling_parse') return '📑';
    if (toolName === 'vector_memory') return '🧠';
    if (toolName === 'file_write') return '✏️';
    if (toolName === 'file_edit') return '📝';
    if (toolName.startsWith('github_')) return '🐙';
    return '⚙️';
  }

  // Tab Switching
  if (tabBtnCatalog && tabBtnTester && tabBtnCheckpoints) {
    tabBtnCatalog.addEventListener('click', () => switchToolsTab('catalog'));
    tabBtnTester.addEventListener('click', () => switchToolsTab('tester'));
    tabBtnCheckpoints.addEventListener('click', () => {
      switchToolsTab('checkpoints');
      fetchCheckpoints();
    });
  }

  function switchToolsTab(tabName) {
    [tabBtnCatalog, tabBtnTester, tabBtnCheckpoints].forEach((btn) => {
      if (!btn) return;
      btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    if (tabPaneCatalog) tabPaneCatalog.style.display = tabName === 'catalog' ? 'flex' : 'none';
    if (tabPaneTester) tabPaneTester.style.display = tabName === 'tester' ? 'flex' : 'none';
    if (tabPaneCheckpoints) tabPaneCheckpoints.style.display = tabName === 'checkpoints' ? 'flex' : 'none';
  }

  async function fetchTools() {
    try {
      const res = await fetch('/api/tools');
      if (!res.ok) return;
      const data = await res.json();
      registeredTools = data.tools || [];
      const count = data.count || registeredTools.length;

      if (toolsBadge) toolsBadge.textContent = count;
      if (topbarToolsBadge) topbarToolsBadge.textContent = count;
      if (tabToolsCount) tabToolsCount.textContent = count;
      if (emptyToolsCount) emptyToolsCount.textContent = count;
      if (inlineToolsBadge) inlineToolsBadge.textContent = count;
      if (catCountAll) catCountAll.textContent = count;

      renderEmptyToolsPills(registeredTools);
      renderToolsCatalog(registeredTools, currentToolsSearch, currentToolsCategory);
      populateTesterSelect(registeredTools);
    } catch {
      if (toolsGrid) {
        toolsGrid.innerHTML = '<div style="color: var(--status-offline); padding: 12px;">Failed to load tools registry.</div>';
      }
    }
  }

  function renderEmptyToolsPills(tools) {
    if (!emptyToolsPills) return;
    if (!tools || tools.length === 0) {
      emptyToolsPills.innerHTML = '<span style="color: var(--text-faint); font-size: 12px;">No tools loaded.</span>';
      return;
    }

    emptyToolsPills.innerHTML = '';
    tools.forEach((t) => {
      const cat = getToolCategory(t.name);
      const icon = getToolIcon(t.name);
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = `empty-tool-chip cat-${cat}`;
      chip.title = `${t.name}: ${t.description || 'Click to test or insert sample prompt'}`;
      chip.innerHTML = `
        <span class="chip-icon">${icon}</span>
        <span class="chip-name">${escapeHtml(t.name)}</span>
      `;

      chip.addEventListener('click', () => {
        // Insert helpful sample prompt in chat textarea
        if (chatTextarea) {
          if (t.name === 'math_eval') {
            chatTextarea.value = 'Calculate (45 * 12) + 8 using math tools and explain the result in one sentence.';
          } else if (t.name === 'web_search') {
            chatTextarea.value = 'Search the web for the latest LangGraph agent architecture in 2026.';
          } else if (t.name === 'python_repl') {
            chatTextarea.value = 'Run Python code to calculate the first 10 Fibonacci numbers.';
          } else if (t.name === 'wikipedia') {
            chatTextarea.value = 'Look up LangGraph and agentic AI on Wikipedia.';
          } else if (t.name === 'arxiv') {
            chatTextarea.value = 'Search ArXiv for recent papers on multi-agent collaboration.';
          } else if (t.name === 'file_write' || t.name === 'file_edit') {
            chatTextarea.value = 'Create a new python file named summary_output.py with a hello world function.';
          } else {
            chatTextarea.value = `Please execute the ${t.name} tool.`;
          }
          chatTextarea.focus();
          adjustTextareaHeight();
          checkSendButtonState();
        }
      });

      emptyToolsPills.appendChild(chip);
    });
  }

  function renderToolsCatalog(tools, filterQuery = '', category = 'all') {
    if (!toolsGrid) return;
    if (!tools || tools.length === 0) {
      toolsGrid.innerHTML = '<div class="loading-spinner">No tools currently registered.</div>';
      return;
    }

    const q = (filterQuery || '').toLowerCase().trim();
    const filtered = tools.filter((t) => {
      const matchCat = category === 'all' || getToolCategory(t.name) === category;
      if (!matchCat) return false;
      if (!q) return true;
      const inName = t.name.toLowerCase().includes(q);
      const inDesc = (t.description || '').toLowerCase().includes(q);
      const inParams = (t.params || []).some((p) => p.toLowerCase().includes(q));
      return inName || inDesc || inParams;
    });

    if (filtered.length === 0) {
      toolsGrid.innerHTML = `<div style="grid-column: 1/-1; color: var(--text-muted); font-size: 13px; padding: 24px; text-align: center;">No tools matching "${escapeHtml(filterQuery)}" in category "${escapeHtml(category)}".</div>`;
      return;
    }

    toolsGrid.innerHTML = '';
    filtered.forEach((t) => {
      const cat = getToolCategory(t.name);
      const catLabel = getToolCategoryLabel(cat);
      const icon = getToolIcon(t.name);
      const card = document.createElement('div');
      card.className = `tool-card tool-cat-${cat}`;

      let paramsHtml = '';
      if (t.params && t.params.length > 0) {
        paramsHtml = `<div class="tool-params-list">`;
        t.params.forEach((p) => {
          paramsHtml += `<span class="tool-param-pill">${escapeHtml(p)}</span>`;
        });
        paramsHtml += `</div>`;
      } else {
        paramsHtml = `<div class="tool-params-list"><span class="tool-param-pill">no params</span></div>`;
      }

      card.innerHTML = `
        <div class="tool-card-top">
          <div class="tool-card-identity">
            <span class="tool-icon-span">${icon}</span>
            <span class="tool-name-badge">${escapeHtml(t.name)}</span>
          </div>
          <span class="tool-category-badge cat-${cat}">${catLabel}</span>
        </div>
        <div class="tool-desc">${escapeHtml(t.description || 'No description provided.')}</div>
        ${paramsHtml}
        <div class="tool-card-actions">
          <button type="button" class="tool-test-btn" data-tool="${escapeHtml(t.name)}">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
            <span>Test in Runner</span>
          </button>
          <button type="button" class="tool-prompt-btn" data-tool="${escapeHtml(t.name)}">Use in Chat</button>
        </div>
      `;

      card.querySelector('.tool-test-btn').addEventListener('click', () => {
        selectToolInTester(t.name);
      });

      card.querySelector('.tool-prompt-btn').addEventListener('click', () => {
        closeToolsModal();
        if (chatTextarea) {
          chatTextarea.value = `Please execute the ${t.name} tool.`;
          chatTextarea.focus();
          adjustTextareaHeight();
          checkSendButtonState();
        }
      });

      toolsGrid.appendChild(card);
    });
  }

  // Search & category filter event listeners
  if (toolsSearchInput) {
    toolsSearchInput.addEventListener('input', (e) => {
      currentToolsSearch = e.target.value;
      renderToolsCatalog(registeredTools, currentToolsSearch, currentToolsCategory);
    });
  }

  if (toolsCategoryFilters) {
    const catButtons = toolsCategoryFilters.querySelectorAll('.tool-cat-btn');
    catButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        catButtons.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        currentToolsCategory = btn.dataset.cat || 'all';
        renderToolsCatalog(registeredTools, currentToolsSearch, currentToolsCategory);
      });
    });
  }

  function populateTesterSelect(tools) {
    if (!testerToolSelect) return;
    testerToolSelect.innerHTML = '';
    tools.forEach((t) => {
      const opt = document.createElement('option');
      opt.value = t.name;
      opt.textContent = `${t.name} (${t.params.join(', ') || 'no params'})`;
      testerToolSelect.appendChild(opt);
    });

    if (tools.length > 0) {
      updateTesterTemplate(tools[0].name);
    }
  }

  function selectToolInTester(toolName) {
    switchToolsTab('tester');
    if (testerToolSelect) {
      testerToolSelect.value = toolName;
      updateTesterTemplate(toolName);
    }
  }

  if (testerToolSelect) {
    testerToolSelect.addEventListener('change', (e) => {
      updateTesterTemplate(e.target.value);
    });
  }

  function updateTesterTemplate(toolName) {
    if (!testerToolArgs) return;
    if (toolName === 'web_search') {
      testerToolArgs.value = JSON.stringify({ query: 'LangGraph architecture and agent tools', max_results: 3 }, null, 2);
    } else if (toolName === 'math_eval') {
      testerToolArgs.value = JSON.stringify({ expression: '125 * 37 - 50' }, null, 2);
    } else if (toolName === 'python_repl') {
      testerToolArgs.value = JSON.stringify({ code: 'import math\nprint(f"pi={math.pi:.4f}, sqrt(2)={math.sqrt(2):.4f}")' }, null, 2);
    } else if (toolName === 'wikipedia') {
      testerToolArgs.value = JSON.stringify({ query: 'LangGraph', sentences: 3 }, null, 2);
    } else if (toolName === 'arxiv') {
      testerToolArgs.value = JSON.stringify({ query: 'Large Language Models multi-agent', max_results: 2 }, null, 2);
    } else if (toolName === 'web_scrape') {
      testerToolArgs.value = JSON.stringify({ url: 'https://example.com' }, null, 2);
    } else if (toolName === 'github_status') {
      testerToolArgs.value = JSON.stringify({ repo_dir: '.' }, null, 2);
    } else {
      const toolObj = registeredTools.find((t) => t.name === toolName);
      if (toolObj && toolObj.params) {
        const dummy = {};
        toolObj.params.forEach((p) => { dummy[p] = 'example_value'; });
        testerToolArgs.value = JSON.stringify(dummy, null, 2);
      } else {
        testerToolArgs.value = '{}';
      }
    }
  }

  if (runToolTesterBtn) {
    runToolTesterBtn.addEventListener('click', async () => {
      const toolName = testerToolSelect.value;
      if (!toolName) return;

      let parsedArgs = {};
      try {
        const raw = testerToolArgs.value.trim();
        if (raw) parsedArgs = JSON.parse(raw);
      } catch (err) {
        alert(`Invalid JSON in Arguments: ${err.message}`);
        return;
      }

      runToolTesterBtn.disabled = true;
      runToolTesterBtn.innerHTML = '<span>Executing...</span>';
      testerOutputBox.style.display = 'block';
      testerOutputContent.textContent = 'Running tool with SQLite checkpointing...';
      testerStatusBadge.className = 'output-status-badge';
      testerStatusBadge.textContent = 'RUNNING';
      testerCheckpointFooter.innerHTML = '';

      const startT = performance.now();
      try {
        const res = await fetch('/api/tools/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            tool: toolName,
            args: parsedArgs,
            metadata: { source: 'web_ui_tester' },
          }),
        });
        const result = await res.json();
        const elapsed = ((performance.now() - startT) / 1000).toFixed(3);

        const ok = result.ok !== false;
        testerStatusBadge.className = `output-status-badge ${ok ? 'success' : 'error'}`;
        testerStatusBadge.textContent = ok ? 'SUCCESS (200)' : 'FAILED';

        const displayMsg = result.message !== undefined ? result.message : JSON.stringify(result, null, 2);
        testerOutputContent.textContent = typeof displayMsg === 'object' ? JSON.stringify(displayMsg, null, 2) : String(displayMsg);

        let footerChips = `<span class="checkpoint-chip">Duration: ${elapsed}s</span>`;
        if (result.checkpoint_id) {
          footerChips += `<span class="checkpoint-chip">Pre-CP: ${escapeHtml(result.checkpoint_id)}</span>`;
        }
        if (result.post_checkpoint_id) {
          footerChips += `<span class="checkpoint-chip">Post-CP: ${escapeHtml(result.post_checkpoint_id)}</span>`;
        }
        testerCheckpointFooter.innerHTML = footerChips;
      } catch (err) {
        testerStatusBadge.className = 'output-status-badge error';
        testerStatusBadge.textContent = 'ERROR';
        testerOutputContent.textContent = `Network / Execution error: ${err.message}`;
      } finally {
        runToolTesterBtn.disabled = false;
        runToolTesterBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg>
          <span>Run Tool with Checkpoint</span>
        `;
      }
    });
  }

  async function fetchCheckpoints(filterQuery) {
    if (!checkpointsList) return;
    checkpointsList.innerHTML = '<div class="loading-spinner">Loading checkpoints from SQLite...</div>';

    try {
      const url = filterQuery ? `/api/tools/checkpoints?limit=50&tool_name=${encodeURIComponent(filterQuery)}` : '/api/tools/checkpoints?limit=50';
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const checkpoints = data.checkpoints || [];

      if (checkpoints.length === 0) {
        checkpointsList.innerHTML = '<div style="color: var(--text-muted); font-size: 13px; padding: 12px;">No tool checkpoints recorded in memory.</div>';
        return;
      }

      checkpointsList.innerHTML = '';
      checkpoints.forEach((cp) => {
        const item = document.createElement('div');
        item.className = 'checkpoint-item';

        const eventName = cp.event || 'checkpoint';
        const toolName = (cp.metadata && cp.metadata.tool) ? cp.metadata.tool : 'tool';
        const type = (cp.metadata && cp.metadata.checkpoint_type) ? cp.metadata.checkpoint_type : 'checkpoint';
        const timeVal = cp.timestamp || 'N/A';
        const resultSummary = cp.result ? String(cp.result).slice(0, 300) : 'No result payload';

        item.innerHTML = `
          <div class="checkpoint-item-top">
            <span class="checkpoint-event-tag">${escapeHtml(eventName)}</span>
            <span class="checkpoint-item-meta">${escapeHtml(timeVal)}</span>
          </div>
          <div class="checkpoint-item-meta">
            <span>Tool: <strong>${escapeHtml(toolName)}</strong></span>
            <span>Type: ${escapeHtml(type)}</span>
            <span>ID: ${escapeHtml(cp.id || 'N/A')}</span>
          </div>
          <div class="checkpoint-item-summary">${escapeHtml(resultSummary)}</div>
        `;
        checkpointsList.appendChild(item);
      });
    } catch {
      checkpointsList.innerHTML = '<div style="color: var(--status-offline); font-size: 13px; padding: 8px;">Failed to load checkpoints.</div>';
    }
  }

  if (refreshCheckpointsBtn) {
    refreshCheckpointsBtn.addEventListener('click', () => {
      const q = checkpointsFilterInput ? checkpointsFilterInput.value.trim() : '';
      fetchCheckpoints(q);
    });
  }

  if (checkpointsFilterInput) {
    let cpFilterTimer = null;
    checkpointsFilterInput.addEventListener('input', () => {
      clearTimeout(cpFilterTimer);
      cpFilterTimer = setTimeout(() => {
        fetchCheckpoints(checkpointsFilterInput.value.trim());
      }, 300);
    });
  }

  // Initial loads
  refreshRunsHistory();
  refreshMemoryMetrics();
  fetchTools();
});

