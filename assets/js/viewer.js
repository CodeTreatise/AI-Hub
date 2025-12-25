        let data = null;
        let simulation = null;
        let currentLayout = 'force';
        let activeSection = null;
        
        // Color palette
        const colors = d3.schemeTableau10;
        
        // Get graph ID from URL
        const urlParams = new URLSearchParams(window.location.search);
        const graphId = urlParams.get('graph') || 'ai-fundamentals';
        
        // Load graph data
        async function loadGraph() {
            try {
                const response = await fetch(`../data/graphs/${graphId}.json`);
                if (!response.ok) throw new Error('Graph not found');
                data = await response.json();
                
                // Update header
                document.getElementById('graph-icon').textContent = data.meta.icon || '📊';
                document.getElementById('graph-title').textContent = data.meta.title;
                document.getElementById('graph-description').textContent = data.meta.description || '';
                document.title = `${data.meta.title} - AI Graph`;
                
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                
                // Render
                renderSections();
                initGraph();
                updateStats();
            } catch (error) {
                console.error('Failed to load graph:', error);
                document.getElementById('loading').innerHTML = `
                    <p style="color: #f85149;">Failed to load graph: ${graphId}</p>
                    <a href="../index.html" style="color: #58a6ff;">← Back to Hub</a>
                `;
            }
        }
        
        // Render sidebar sections
        function renderSections() {
            const container = document.getElementById('sections-list');
            if (!data.sections) {
                container.innerHTML = '<div style="padding:1rem; color:#8b949e; font-size:0.9rem;">No sections defined</div>';
                return;
            }
            container.innerHTML = data.sections.map((section, i) => {
                const nodeCount = data.nodes.filter(n => n.section_id === section.id).length;
                return `
                    <div class="section-item" data-id="${section.id}" onclick="toggleSection(${section.id})">
                        <div class="color-dot" style="background: ${colors[i % colors.length]}"></div>
                        <div class="section-text">
                            <div class="section-title">${section.title}</div>
                            <div class="section-count">${nodeCount} nodes</div>
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        // Toggle section highlight
        function toggleSection(sectionId) {
            const items = document.querySelectorAll('.section-item');
            items.forEach(item => {
                const id = parseInt(item.dataset.id);
                item.classList.toggle('active', id === sectionId && activeSection !== sectionId);
            });
            
            activeSection = activeSection === sectionId ? null : sectionId;
            updateNodeVisibility();
        }
        
        // Update node visibility based on active section
        function updateNodeVisibility() {
            const nodes = d3.selectAll('.node');
            const links = d3.selectAll('.link');
            const labels = d3.selectAll('.node-label');
            
            if (activeSection === null) {
                nodes.attr('opacity', 1);
                links.attr('opacity', 0.3);
                labels.attr('opacity', 1);
            } else {
                const sectionNodeIds = new Set(
                    data.nodes.filter(n => n.section_id === activeSection).map(n => n.id)
                );
                
                nodes.attr('opacity', d => sectionNodeIds.has(d.id) ? 1 : 0.15);
                links.attr('opacity', d => 
                    sectionNodeIds.has(d.source.id) || sectionNodeIds.has(d.target.id) ? 0.5 : 0.05
                );
                labels.attr('opacity', d => sectionNodeIds.has(d.id) ? 1 : 0.15);
            }
        }
        
        // Initialize graph
        function initGraph() {
            const svg = d3.select('#graph-svg');
            const container = document.getElementById('graph-container');
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            svg.selectAll('*').remove();
            
            const g = svg.append('g');
            
            // Create node/edge data with indices
            const nodeMap = new Map(data.nodes.map((n, i) => [n.id, i]));
            const nodes = data.nodes.map(n => ({...n}));
            const links = data.edges
                .filter(e => nodeMap.has(e.source) && nodeMap.has(e.target))
                .map(e => ({
                    source: e.source,
                    target: e.target
                }));
            
            // Section colors
            const sectionIndex = data.sections 
                ? new Map(data.sections.map((s, i) => [s.id, i]))
                : new Map();
            
            // Draw links
            const link = g.append('g')
                .selectAll('line')
                .data(links)
                .join('line')
                .attr('class', 'link')
                .attr('stroke', '#30363d')
                .attr('stroke-opacity', 0.3)
                .attr('stroke-width', 1);
            
            // Draw nodes
            const node = g.append('g')
                .selectAll('circle')
                .data(nodes)
                .join('circle')
                .attr('class', 'node')
                .attr('r', d => d.type === 'subgraph' ? 8 : 5)
                .attr('fill', d => {
                    if (d.section_id && sectionIndex.has(d.section_id)) {
                        return colors[sectionIndex.get(d.section_id) % colors.length];
                    }
                    if (d.level !== undefined) {
                        return colors[d.level % colors.length];
                    }
                    return '#888';
                })
                .attr('stroke', '#0d1117')
                .attr('stroke-width', 1.5)
                .style('cursor', 'pointer')
                .on('mouseover', showTooltip)
                .on('mouseout', hideTooltip)
                .on('click', (e, d) => {
                    if (d.local_path) window.open('../' + d.local_path, '_blank');
                    else if (d.url) window.open(d.url, '_blank');
                })
                .call(d3.drag()
                    .on('start', dragStart)
                    .on('drag', dragging)
                    .on('end', dragEnd));
            
            // Draw labels
            const labels = g.append('g')
                .selectAll('text')
                .data(nodes)
                .join('text')
                .attr('class', 'node-label')
                .attr('font-size', '9px')
                .attr('fill', '#c9d1d9')
                .attr('text-anchor', 'middle')
                .attr('dy', d => d.type === 'subgraph' ? 18 : 14)
                .attr('pointer-events', 'none')
                .text(d => {
                    // Clean up label - remove emoji prefixes and shorten
                    let label = d.label.replace(/^[\u{1F300}-\u{1F9FF}]+ ?/u, '');
                    return label.length > 18 ? label.slice(0, 16) + '...' : label;
                });
            
            // Force simulation
            simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links).id(d => d.id).distance(80))
                .force('charge', d3.forceManyBody().strength(-200))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(30))
                .on('tick', () => {
                    link
                        .attr('x1', d => d.source.x)
                        .attr('y1', d => d.source.y)
                        .attr('x2', d => d.target.x)
                        .attr('y2', d => d.target.y);
                    
                    node
                        .attr('cx', d => d.x)
                        .attr('cy', d => d.y);
                    
                    labels
                        .attr('x', d => d.x)
                        .attr('y', d => d.y);
                });
            
            // Zoom
            const zoom = d3.zoom()
                .scaleExtent([0.1, 4])
                .on('zoom', e => g.attr('transform', e.transform));
            
            svg.call(zoom);
            
            // Reset zoom button
            document.getElementById('reset-zoom').onclick = () => {
                svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
            };
            
            // Toggle labels button
            document.getElementById('toggle-labels').onclick = function() {
                this.classList.toggle('active');
                document.getElementById('graph-svg').classList.toggle('labels-hidden');
            };
            
            // Layout buttons
            document.querySelectorAll('[data-layout]').forEach(btn => {
                btn.onclick = () => {
                    document.querySelectorAll('[data-layout]').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    currentLayout = btn.dataset.layout;
                    applyLayout(nodes, width, height);
                };
            });
            
            // Store references
            window.graphRefs = { nodes, links, node, link, labels, simulation, width, height };
        }
        
        // Apply layout
        function applyLayout(nodes, width, height) {
            const { simulation } = window.graphRefs;
            
            simulation.stop();
            
            if (currentLayout === 'force') {
                simulation
                    .force('center', d3.forceCenter(width / 2, height / 2))
                    .force('charge', d3.forceManyBody().strength(-100))
                    .alpha(1)
                    .restart();
            } else if (currentLayout === 'radial') {
                const sectionGroups = d3.group(nodes, d => d.section_id);
                const sectionIds = [...sectionGroups.keys()];
                const angleStep = (2 * Math.PI) / sectionIds.length;
                
                sectionIds.forEach((sectionId, i) => {
                    const angle = i * angleStep - Math.PI / 2;
                    const radius = 200;
                    const cx = width / 2 + Math.cos(angle) * radius;
                    const cy = height / 2 + Math.sin(angle) * radius;
                    
                    sectionGroups.get(sectionId).forEach((node, j) => {
                        const subAngle = (j / sectionGroups.get(sectionId).length) * Math.PI * 0.5 + angle - Math.PI * 0.25;
                        const subRadius = 60;
                        node.x = cx + Math.cos(subAngle) * subRadius;
                        node.y = cy + Math.sin(subAngle) * subRadius;
                    });
                });
                
                simulation.alpha(0.3).restart();
            } else if (currentLayout === 'cluster') {
                const sectionGroups = d3.group(nodes, d => d.section_id);
                const cols = Math.ceil(Math.sqrt(sectionGroups.size));
                const cellW = width / cols;
                const cellH = height / Math.ceil(sectionGroups.size / cols);
                
                let i = 0;
                sectionGroups.forEach((group, sectionId) => {
                    const col = i % cols;
                    const row = Math.floor(i / cols);
                    const cx = col * cellW + cellW / 2;
                    const cy = row * cellH + cellH / 2;
                    
                    group.forEach((node, j) => {
                        const angle = (j / group.length) * 2 * Math.PI;
                        const radius = Math.min(cellW, cellH) * 0.3;
                        node.x = cx + Math.cos(angle) * radius * (0.5 + Math.random() * 0.5);
                        node.y = cy + Math.sin(angle) * radius * (0.5 + Math.random() * 0.5);
                    });
                    i++;
                });
                
                simulation.alpha(0.3).restart();
            }
        }
        
        // Tooltip
        function showTooltip(event, d) {
            const tooltip = document.getElementById('tooltip');
            const section = data.sections.find(s => s.id === d.section_id);
            
            tooltip.innerHTML = `
                <div class="tooltip-title">${d.label}</div>
                <div class="tooltip-section">${section?.title || 'Unknown'}</div>
                ${d.url ? `<div class="tooltip-url">${d.url}</div>` : ''}
            `;
            
            tooltip.style.left = (event.pageX + 15) + 'px';
            tooltip.style.top = (event.pageY - 10) + 'px';
            tooltip.classList.add('visible');
        }
        
        function hideTooltip() {
            document.getElementById('tooltip').classList.remove('visible');
        }
        
        // Drag handlers
        function dragStart(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragging(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragEnd(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        
        // Update stats
        function updateStats() {
            const sectionCount = data.sections ? data.sections.length : 0;
            document.getElementById('stats').innerHTML = `
                <span>${data.nodes.length}</span> nodes · 
                <span>${data.edges.length}</span> edges · 
                <span>${sectionCount}</span> sections
            `;
        }
        
        // Initialize
        loadGraph();
        
        // Handle resize
        window.addEventListener('resize', () => {
            if (data) initGraph();
        });
