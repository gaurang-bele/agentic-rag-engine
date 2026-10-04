document.addEventListener('DOMContentLoaded', () => {
  // 3D PERSPECTIVE TILT HANDLER
  const rm = matchMedia('(prefers-reduced-motion:reduce)').matches;
  function applyTilt(card) {
    if (rm || !card) return;
    card.addEventListener('pointermove', e => {
      const r = card.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width - 0.5;
      const y = (e.clientY - r.top) / r.height - 0.5;
      card.style.transform = `perspective(700px) rotateX(${-y * 8}deg) rotateY(${x * 10}deg)`;
    });
    card.addEventListener('pointerleave', () => {
      card.style.transform = '';
    });
  }

  document.querySelectorAll('.panel-card, .main-tabs, .analytics-ticker-bar').forEach(applyTilt);

  // LIGHTWEIGHT 60FPS CANVAS PARTICLE BG
  const cv = document.getElementById('bg');
  if (cv) {
    const ctx = cv.getContext('2d');
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const sp = document.createElement('canvas');
    sp.width = sp.height = 32;
    const sc = sp.getContext('2d');
    const g = sc.createRadialGradient(16, 16, 0, 16, 16, 16);
    g.addColorStop(0, 'rgba(255, 255, 255, 0.95)');
    g.addColorStop(0.25, 'rgba(34, 211, 238, 0.55)');
    g.addColorStop(1, 'rgba(34, 211, 238, 0)');
    sc.fillStyle = g;
    sc.fillRect(0, 0, 32, 32);

    let W, H, N = [], px = 0, tx = 0;
    function resizeCanvas() {
      W = window.innerWidth;
      H = window.innerHeight;
      cv.width = W * dpr;
      cv.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      N = Array.from({ length: Math.min(30, Math.round(W / 40)) }, () => ({
        x: Math.random() * W,
        y: Math.random() * H,
        s: Math.random() * 12 + 6,
        v: Math.random() * 0.3 + 0.1,
        p: Math.random() * 6.28,
        k: Math.random() * 0.02 + 0.01
      }));
    }

    function drawParticles() {
      ctx.clearRect(0, 0, W, H);
      px += (tx - px) * 0.05;
      for (const a of N) {
        a.y -= a.v;
        a.p += a.k;
        if (a.y < -20) {
          a.y = H + 20;
          a.x = Math.random() * W;
        }
        ctx.globalAlpha = 0.2 + 0.5 * Math.abs(Math.sin(a.p));
        ctx.drawImage(sp, a.x + Math.sin(a.p) * 12 + px * a.s * 0.08, a.y, a.s * 2, a.s * 2);
      }
      ctx.globalAlpha = 1;
    }

    function animLoop() {
      drawParticles();
      if (!rm) requestAnimationFrame(animLoop);
    }

    window.addEventListener('pointermove', e => {
      tx = (e.clientX / window.innerWidth - 0.5) * 30;
    });

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    animLoop();
  }

  // TAB SWITCHING
  const tabItems = document.querySelectorAll('.tab-item');
  const tabPages = document.querySelectorAll('.tab-page');

  tabItems.forEach(tab => {
    tab.addEventListener('click', () => {
      tabItems.forEach(t => t.classList.remove('active'));
      tabPages.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetId = tab.getAttribute('data-tab');
      document.getElementById(targetId).classList.add('active');
    });
  });


  // QUICK CHIP PROMPTS (Query)
  const chipBtns = document.querySelectorAll('.chip-btn');
  const queryInput = document.getElementById('query-input');

  chipBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      queryInput.value = btn.getAttribute('data-prompt');
    });
  });

  // QUICK AGENT CHIPS
  const agentChipBtns = document.querySelectorAll('.agent-chip-btn');
  const agentInput = document.getElementById('agent-input');

  agentChipBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      agentInput.value = btn.getAttribute('data-agent-prompt');
    });
  });

  // COPY ANSWER BUTTON
  const btnCopyAnswer = document.getElementById('btn-copy-answer');
  const queryAnswerDisplay = document.getElementById('query-answer-display');

  if (btnCopyAnswer) {
    btnCopyAnswer.addEventListener('click', () => {
      const text = queryAnswerDisplay.textContent;
      if (!text) return;
      navigator.clipboard.writeText(text);
      btnCopyAnswer.textContent = '✅ Copied!';
      setTimeout(() => {
        btnCopyAnswer.textContent = '📋 Copy Answer';
      }, 2000);
    });
  }

  // TAB 1: INGESTION DROPZONE & UPLOAD
  const dropzone = document.getElementById('pdf-dropzone');
  const fileInput = document.getElementById('pdf-file-input');
  const fileInfoBadge = document.getElementById('file-info-badge');
  const btnIngest = document.getElementById('btn-ingest');
  const parserSelect = document.getElementById('parser-type-select');
  const progressBox = document.getElementById('ingest-progress-box');
  const progressFill = document.getElementById('progress-bar-fill');
  const progressStatus = document.getElementById('progress-status-text');
  const progressPercent = document.getElementById('progress-percent-text');

  let selectedFile = null;

  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = '#06b6d4';
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = 'rgba(6, 182, 212, 0.35)';
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'rgba(6, 182, 212, 0.35)';
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      return alert('Please select a PDF document.');
    }
    selectedFile = file;
    fileInfoBadge.textContent = `📄 Selected File: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    fileInfoBadge.classList.remove('hidden');
    btnIngest.disabled = false;
  }

  btnIngest.addEventListener('click', async () => {
    if (!selectedFile) return;

    btnIngest.disabled = true;
    btnIngest.textContent = 'Ingesting PDF...';
    progressBox.classList.remove('hidden');
    updateProgress(15, 'Uploading PDF to server...');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      updateProgress(45, 'Parsing document with LlamaParse Markdown layout engine...');
      const url = `/ingest?parser_type=${parserSelect.value}&wait=true`;
      const res = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Ingestion failed');
      }

      const data = await res.json();
      updateProgress(100, `Completed! Ingested ${data.result?.chunks_count || 0} chunks.`);
    } catch (err) {
      updateProgress(100, 'Ingestion Error');
      alert('Ingestion error: ' + err.message);
    } finally {
      btnIngest.disabled = false;
      btnIngest.textContent = '🚀 Start Document Ingestion';
    }
  });

  function updateProgress(percent, msg) {
    progressFill.style.width = percent + '%';
    progressPercent.textContent = percent + '%';
    progressStatus.textContent = msg;
  }

  // TAB 2: QUERY ENGINE
  const btnQuery = document.getElementById('btn-query');
  const querySource = document.getElementById('query-source');
  const queryTopk = document.getElementById('query-topk');
  const queryRerank = document.getElementById('query-rerank');
  const queryResultContainer = document.getElementById('query-result-container');
  const querySourcesGrid = document.getElementById('query-sources-grid');

  btnQuery.addEventListener('click', async () => {
    const question = queryInput.value.trim();
    if (!question) return alert('Please enter a question.');

    btnQuery.disabled = true;
    btnQuery.textContent = 'Searching...';
    queryResultContainer.classList.add('hidden');

    try {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question,
          source: querySource.value.trim() || null,
          top_k: parseInt(queryTopk.value) || 5,
          rerank: queryRerank.checked,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Query failed');
      }
      const data = await res.json();
      queryAnswerDisplay.textContent = data.answer || 'No answer returned.';

      updateTickerMetrics(data);

      window.lastQueryResult = {
        title: "RAG Query Report",
        query: question,
        answer: data.answer,
        sources: data.sources || [],
      };

      querySourcesGrid.innerHTML = '';
      if (data.sources && data.sources.length > 0) {
        data.sources.forEach((src, idx) => {
          const item = document.createElement('div');
          item.className = 'source-item-card';
          const score = src.rerank_score !== undefined && src.rerank_score !== null
            ? `BGE Score: ${src.rerank_score}`
            : 'Similarity Chunk';
          item.innerHTML = `
            <div class="source-item-header">
              <span>[Chunk ${idx + 1}] ${src.source || 'Doc'} (Page ${src.page || '?'})</span>
              <span>${score}</span>
            </div>
            <div class="source-item-body">${escapeHtml(src.content)}...</div>
          `;
          querySourcesGrid.appendChild(item);
        });
      }

      queryResultContainer.classList.remove('hidden');
    } catch (err) {
      alert('Query error: ' + err.message);
    } finally {
      btnQuery.disabled = false;
      btnQuery.textContent = 'Search & Answer';
    }
  });

  // TAB 3: AGENT
  const btnAgent = document.getElementById('btn-agent');
  const agentResultContainer = document.getElementById('agent-result-container');
  const agentOutputDisplay = document.getElementById('agent-output-display');
  const agentStepsTimeline = document.getElementById('agent-steps-timeline');

  const searchDepthSelect = document.getElementById('search-depth-select');

  btnAgent.addEventListener('click', async () => {
    let input = agentInput.value.trim();
    if (!input) return alert('Please enter an agent instruction.');

    if (searchDepthSelect && searchDepthSelect.value === 'advanced') {
      input += ' (Note: use search_depth=advanced for web search tool if needed)';
    }

    btnAgent.disabled = true;
    btnAgent.textContent = 'Executing Agent...';
    agentResultContainer.classList.add('hidden');

    try {
      const res = await fetch('/agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: input }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Agent execution failed');
      }

      const data = await res.json();
      agentOutputDisplay.textContent = data.output || 'Agent task completed.';

      updateTickerMetrics(data);

      window.lastAgentResult = {
        title: "Agent Autonomous Task Report",
        query: input,
        answer: data.output,
        steps: data.intermediate_steps || [],
      };


      agentStepsTimeline.innerHTML = '';
      if (data.intermediate_steps && data.intermediate_steps.length > 0) {
        data.intermediate_steps.forEach((step, idx) => {
          const div = document.createElement('div');
          div.className = 'source-item-card';
          div.innerHTML = `
            <div class="source-item-header">
              <span>Step ${idx + 1}: Tool '${step.tool}'</span>
            </div>
            <div class="source-item-body">
              <strong>Tool Input:</strong> ${escapeHtml(JSON.stringify(step.tool_input))}<br>
              <strong>Observation:</strong> ${escapeHtml(String(step.observation).slice(0, 350))}...
            </div>
          `;
          agentStepsTimeline.appendChild(div);
        });
      }

      agentResultContainer.classList.remove('hidden');
    } catch (err) {
      alert('Agent error: ' + err.message);
    } finally {
      btnAgent.disabled = false;
      btnAgent.textContent = 'Execute Agent';
    }
  });

  // EXPORT REPORT HANDLERS
  async function downloadReport(payload, format) {
    if (!payload || !payload.answer) return alert('No active result to export.');
    payload.format = format;

    try {
      const res = await fetch('/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Export failed');
      }

      const data = await res.json();
      const link = document.createElement('a');
      link.href = data.download_url;
      link.download = data.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      alert(`✅ Report exported successfully as ${data.filename}!\nSaved locally to outputs/ and uploaded to S3.`);
    } catch (err) {
      alert('Export error: ' + err.message);
    }
  }

  document.getElementById('btn-export-query-md')?.addEventListener('click', () => downloadReport(window.lastQueryResult, 'markdown'));
  document.getElementById('btn-export-query-pdf')?.addEventListener('click', () => downloadReport(window.lastQueryResult, 'pdf'));
  document.getElementById('btn-export-agent-md')?.addEventListener('click', () => downloadReport(window.lastAgentResult, 'markdown'));
  document.getElementById('btn-export-agent-pdf')?.addEventListener('click', () => downloadReport(window.lastAgentResult, 'pdf'));

  const tickerLatency = document.getElementById('ticker-latency');
  const tickerBge = document.getElementById('ticker-bge');
  const tickerCost = document.getElementById('ticker-cost');

  function updateTickerMetrics(data) {
    if (!data) return;
    if (data.latency_sec !== undefined && tickerLatency) {
      tickerLatency.textContent = `${data.latency_sec}s`;
    }
    if (tickerBge) {
      if (data.top_rerank_score !== undefined && data.top_rerank_score !== null) {
        tickerBge.textContent = `${data.top_rerank_score}`;
      } else {
        tickerBge.textContent = `N/A`;
      }
    }
    if (data.est_cost_usd !== undefined && tickerCost) {
      tickerCost.textContent = `$${data.est_cost_usd.toFixed(5)}`;
    }
  }


  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
});
