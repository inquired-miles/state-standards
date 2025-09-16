(function() {
    // Global variables
    let currentData = {
        scatter: null,
        heatmap: null,
        themes: null
    };

    function getEndpoint(name) {
        if (window.dashboardEmbeddings && typeof window.dashboardEmbeddings.getEndpoint === 'function') {
            return window.dashboardEmbeddings.getEndpoint(name) || '';
        }
        return '';
    }

    // Preset configurations
    const presets = {
        'broad-overview': {
            clusterSize: 8,
            epsilon: 0.5,
            nNeighbors: 30,
            description: "Shows major educational themes across states"
        },
        'detailed-analysis': {
            clusterSize: 3,
            epsilon: 0.2,
            nNeighbors: 15,
            description: "Finds very specific similarities between standards"
        },
        'subject-focused': {
            clusterSize: 5,
            epsilon: 0.3,
            nNeighbors: 20,
            subjectFocus: 'prefer-same',
            description: "Groups standards within same subject areas"
        }
    };

    // Apply preset function
    window.applyPreset = function(presetType) {
        const preset = presets[presetType];
        if (!preset) return;
        
        // Update slider values
        document.getElementById('cluster-size').value = preset.clusterSize;
        document.getElementById('epsilon').value = preset.epsilon;
        document.getElementById('n-neighbors').value = preset.nNeighbors;
        
        // Update display values
        updateSliderDisplays();
        
        // Update advanced settings if available
        if (preset.subjectFocus) {
            document.getElementById('subject-focus').value = preset.subjectFocus;
        }
        
        // Update impact indicators
        updateParameterImpact();
        
        // Show success message
        showToast(`Applied ${presetType.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}: ${preset.description}`);
        
        // Auto-update visualization
        document.getElementById('update-viz').click();
    };

    // Update slider display values
    function updateSliderDisplays() {
        const clusterSizeValue = document.getElementById('cluster-size-value');
        const epsilonValue = document.getElementById('epsilon-value');
        const neighborsValue = document.getElementById('n-neighbors-value');
        const clusterSizeInput = document.getElementById('cluster-size');
        const epsilonInput = document.getElementById('epsilon');
        const neighborsInput = document.getElementById('n-neighbors');

        if (clusterSizeValue && clusterSizeInput) {
            clusterSizeValue.textContent = clusterSizeInput.value;
        }
        if (epsilonValue && epsilonInput) {
            epsilonValue.textContent = epsilonInput.value;
        }
        if (neighborsValue && neighborsInput) {
            neighborsValue.textContent = neighborsInput.value;
        }
    }

    // Update parameter impact indicators
    function updateParameterImpact() {
        const clusterSizeInput = document.getElementById('cluster-size');
        const epsilonInput = document.getElementById('epsilon');
        const neighborsInput = document.getElementById('n-neighbors');
        const methodSelect = document.getElementById('clustering-method');
        const subjectSelect = document.getElementById('subject-focus');
        if (!clusterSizeInput || !epsilonInput || !neighborsInput || !methodSelect || !subjectSelect) {
            return;
        }

        const clusterSize = parseInt(clusterSizeInput.value);
        const epsilon = parseFloat(epsilonInput.value);
        const nNeighbors = parseInt(neighborsInput.value);
        const clusteringMethod = methodSelect.value;
        const subjectFocus = subjectSelect.value;
        
        // Calculate overall clustering tightness considering all factors
        let tightness = calculateOverallTightness(clusterSize, epsilon, nNeighbors, clusteringMethod, subjectFocus);
        updateTightnessIndicator(tightness);
        
        // Show warnings for problematic parameter combinations
        checkParameterInteractions(clusterSize, epsilon, nNeighbors);
    }
    
    function calculateOverallTightness(clusterSize, epsilon, nNeighbors, method, subjectFocus) {
        // Convert parameters to tightness contributions (0 = very permissive, 1 = very restrictive)
        
        // 1. Similarity Strictness (epsilon) - primary factor
        let epsilonTightness = 1 - epsilon; // Lower epsilon = higher tightness
        
        // 2. Minimum Cluster Size - affects how easily clusters form
        // Smaller cluster size = more permissive (easier to form clusters)
        // Larger cluster size = more restrictive (harder to form clusters)
        let clusterSizeTightness;
        if (clusterSize <= 3) clusterSizeTightness = 0.1; // Very permissive
        else if (clusterSize <= 5) clusterSizeTightness = 0.3; // Permissive
        else if (clusterSize <= 8) clusterSizeTightness = 0.5; // Moderate
        else if (clusterSize <= 12) clusterSizeTightness = 0.7; // Restrictive  
        else clusterSizeTightness = 0.9; // Very restrictive
        
        // 3. Neighborhood Size - affects pattern detection scope
        // Smaller neighborhood = more local/tight patterns
        // Larger neighborhood = more global/loose patterns
        let neighborTightness;
        if (nNeighbors <= 10) neighborTightness = 0.8; // Local = tighter
        else if (nNeighbors <= 20) neighborTightness = 0.5; // Balanced
        else if (nNeighbors <= 35) neighborTightness = 0.3; // Global = looser
        else neighborTightness = 0.1; // Very global = very loose
        
        // 4. Clustering Method adjustment
        let methodTightness = 0.5; // Default baseline
        switch(method) {
            case 'similarity': methodTightness = 0.7; break; // Tends to be more restrictive
            case 'hierarchical': methodTightness = 0.4; break; // Slightly more permissive
            default: methodTightness = 0.5; // HDBSCAN baseline
        }
        
        // 5. Subject Focus adjustment
        let subjectTightness = 0.5; // Default
        switch(subjectFocus) {
            case 'prefer-same': subjectTightness = 0.6; break; // Slightly more restrictive
            case 'separate-subjects': subjectTightness = 0.7; break; // More restrictive
            default: subjectTightness = 0.5; // Mixed baseline
        }
        
        // Weighted combination - epsilon is most important, then cluster size
        let overallTightness = (
            epsilonTightness * 0.4 +        // 40% weight - primary factor
            clusterSizeTightness * 0.3 +     // 30% weight - very important
            neighborTightness * 0.2 +        // 20% weight - moderate impact  
            methodTightness * 0.05 +         // 5% weight - minor adjustment
            subjectTightness * 0.05          // 5% weight - minor adjustment
        );
        
        return Math.max(0, Math.min(1, overallTightness));
    }
    
    function updateTightnessIndicator(tightness) {
        let tightnessText, width, colorClass;
        
        // More nuanced tightness levels with better descriptions
        if (tightness >= 0.8) {
            tightnessText = "Very Restrictive";
            width = "15%";
            colorClass = "bg-danger"; // Might prevent clustering
        } else if (tightness >= 0.7) {
            tightnessText = "Restrictive";
            width = "25%";
            colorClass = "bg-warning";
        } else if (tightness >= 0.5) {
            tightnessText = "Moderate";
            width = "45%";
            colorClass = "bg-success"; // Optimal balance
        } else if (tightness >= 0.3) {
            tightnessText = "Permissive";
            width = "65%";
            colorClass = "bg-info";
        } else if (tightness >= 0.15) {
            tightnessText = "Very Permissive";
            width = "85%";
            colorClass = "bg-warning"; // Might group unrelated concepts
        } else {
            tightnessText = "Extremely Permissive";
            width = "95%";
            colorClass = "bg-danger"; // Likely to group everything
        }
        
        const indicator = document.getElementById('tightness-indicator');
        if (!indicator) {
            return;
        }
        indicator.textContent = tightnessText;
        indicator.style.width = width;
        indicator.className = `progress-bar ${colorClass}`;
    }
    
    function checkParameterInteractions(clusterSize, epsilon, nNeighbors) {
        const warningContainer = document.getElementById('parameter-warnings') || createWarningContainer();
        if (!warningContainer) {
            return;
        }
        warningContainer.innerHTML = ''; // Clear previous warnings
        
        // Check for problematic combinations
        if (clusterSize <= 3 && epsilon >= 0.7) {
            addWarning(warningContainer, 'warning', 'Small clusters + loose similarity may create many tiny, unrelated groups');
        }
        
        if (clusterSize >= 15 && epsilon <= 0.2) {
            addWarning(warningContainer, 'info', 'Large clusters + strict similarity may prevent any clustering');
        }
        
        if (nNeighbors >= 40 && epsilon <= 0.3) {
            addWarning(warningContainer, 'info', 'Large neighborhood + strict similarity may be too conservative');
        }
        
        if (epsilon >= 0.8) {
            addWarning(warningContainer, 'warning', 'Very loose similarity may group unrelated educational concepts');
        }
    }
    
    function createWarningContainer() {
        const container = document.createElement('div');
        container.id = 'parameter-warnings';
        container.className = 'mt-2';
        const panel = document.querySelector('.parameter-impact-panel');
        if (!panel) {
            return null;
        }
        panel.appendChild(container);
        return container;
    }
    
    function addWarning(container, type, message) {
        if (!container) {
            return;
        }
        const alertClass = type === 'warning' ? 'alert-warning' : 'alert-info';
        const icon = type === 'warning' ? 'fa-exclamation-triangle' : 'fa-info-circle';
        
        const warning = document.createElement('div');
        warning.className = `alert ${alertClass} alert-sm py-2 mb-1`;
        warning.innerHTML = `<i class="fas ${icon} me-2"></i><small>${message}</small>`;
        container.appendChild(warning);
    }

    // Show toast notification
    function showToast(message) {
        // Create toast element if it doesn't exist
        let toastContainer = document.getElementById('toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toast-container';
            toastContainer.className = 'position-fixed top-0 end-0 p-3';
            toastContainer.style.zIndex = '1050';
            document.body.appendChild(toastContainer);
        }
        
        const toastId = 'toast-' + Date.now();
        const toastHtml = `
            <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="toast-header">
                    <i class="fas fa-check-circle text-success me-2"></i>
                    <strong class="me-auto">Settings Applied</strong>
                    <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>
        `;
        
        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        const toast = new bootstrap.Toast(document.getElementById(toastId));
        toast.show();
        
        // Remove toast element after it's hidden
        document.getElementById(toastId).addEventListener('hidden.bs.toast', function() {
            this.remove();
        });
    }

    // Initialize tooltips
    function initializeEmbeddingsUI() {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        const tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
        
        const clusterSizeInput = document.getElementById('cluster-size');
        const epsilonInput = document.getElementById('epsilon');
        const neighborsInput = document.getElementById('n-neighbors');
        const clusteringMethod = document.getElementById('clustering-method');
        const subjectFocus = document.getElementById('subject-focus');

        if (clusterSizeInput && epsilonInput && neighborsInput) {
            // Initialize slider displays
            updateSliderDisplays();
            updateParameterImpact();

            clusterSizeInput.addEventListener('input', () => {
                updateSliderDisplays();
                updateParameterImpact();
            });
            epsilonInput.addEventListener('input', () => {
                updateSliderDisplays();
                updateParameterImpact();
            });
            neighborsInput.addEventListener('input', () => {
                updateSliderDisplays();
                updateParameterImpact();
            });
        }

        if (clusteringMethod) {
            clusteringMethod.addEventListener('change', updateParameterImpact);
        }
        if (subjectFocus) {
            subjectFocus.addEventListener('change', updateParameterImpact);
        }
        
        // Add event listeners for network graph filters
        const commonCheckbox = document.getElementById('show-common-concepts');
        const semiCommonCheckbox = document.getElementById('show-semi-common-concepts');
        const stateSpecificCheckbox = document.getElementById('show-state-specific');
        const standardsCheckbox = document.getElementById('show-standards');
        
        if (commonCheckbox) {
            commonCheckbox.addEventListener('change', handleNetworkFilterChange);
        }
        if (semiCommonCheckbox) {
            semiCommonCheckbox.addEventListener('change', handleNetworkFilterChange);
        }
        if (stateSpecificCheckbox) {
            stateSpecificCheckbox.addEventListener('change', handleNetworkFilterChange);
        }
       if (standardsCheckbox) {
           standardsCheckbox.addEventListener('change', handleNetworkFilterChange);
       }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeEmbeddingsUI);
    } else {
        initializeEmbeddingsUI();
    }
    
    // Utility functions
    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) return parts.pop().split(';').shift();
    }
    
    function showLoading(elementId) {
        document.getElementById(elementId).classList.remove('d-none');
    }
    
    function hideLoading(elementId) {
        document.getElementById(elementId).classList.add('d-none');
    }
    
    function getFormData() {
        const gradeLevel = document.getElementById('grade-level');
        const subjectArea = document.getElementById('subject-area');
        const clusterSize = document.getElementById('cluster-size');
        const epsilon = document.getElementById('epsilon');
        const vizMode = document.getElementById('viz-mode');

        return {
            grade_level: gradeLevel ? gradeLevel.value : '',
            subject_area: subjectArea ? subjectArea.value : '',
            cluster_size: clusterSize ? clusterSize.value : '',
            epsilon: epsilon ? epsilon.value : '',
            viz_mode: vizMode ? vizMode.value : '2d'
        };
    }
    
    // Scatter plot visualization
    function renderScatterPlot(data) {
        const container = document.getElementById('scatter-plot');
        container.innerHTML = '';
        
        console.log('renderScatterPlot called with data:', data);
        
        // Validate data structure
        if (!data) {
            container.innerHTML = '<div class="alert alert-warning">No data provided for visualization.</div>';
            return;
        }
        
        // Check if this is 3D mode
        const is3D = data.parameters && data.parameters.viz_mode === '3d';
        console.log('Visualization mode:', is3D ? '3D' : '2D');
        
        // Route to appropriate visualization based on mode
        if (is3D) {
            renderScatterPlot3D(data);
            return;
        }
        
        // Continue with 2D D3.js visualization for 2D mode
        renderScatterPlot2D(data);
    }
    
    // 2D scatter plot using D3.js
    function renderScatterPlot2D(data) {
        const container = document.getElementById('scatter-plot');
        if (!container) {
            console.warn('renderScatterPlot2D: scatter container not found');
            return;
        }

        if (!container.style.minHeight) {
            container.style.minHeight = '420px';
        }

        if (!data.scatter_data || !Array.isArray(data.scatter_data)) {
            container.innerHTML = '<div class="alert alert-warning">Invalid data format: missing scatter_data array.</div>';
            return;
        }
        
        if (data.scatter_data.length === 0) {
            container.innerHTML = `
                <div class="alert alert-info">
                    <h6><i class="fas fa-info-circle"></i> No Standards Found</h6>
                    <p>No standards with embeddings match your current filter criteria.</p>
                    <small class="text-muted">Try:</small>
                    <ul class="small text-muted mt-1 mb-0">
                        <li>Selecting a different grade level or subject area</li>
                        <li>Removing some filters to broaden the search</li>
                        <li>Checking if embeddings have been generated for this data</li>
                    </ul>
                </div>
            `;
            return;
        }
        
        // Comprehensive data validation
        console.log('Raw scatter_data:', data.scatter_data);
        console.log('Raw scatter_data length:', data.scatter_data.length);
        
        // First pass - filter out null/undefined elements
        const nonNullData = data.scatter_data.filter(d => d != null);
        console.log(`After null filter: ${nonNullData.length}/${data.scatter_data.length}`);
        
        // Second pass - validate required properties and fix missing data
        const validDataPoints = nonNullData.map((d, index) => {
            // Check basic requirements
            if (!d || typeof d !== 'object' ||
                typeof d.x !== 'number' || typeof d.y !== 'number' ||
                isNaN(d.x) || isNaN(d.y)) {
                console.warn(`Invalid coordinates at index ${index}:`, d);
                return null;
            }
            
            // Fix missing or empty properties with fallbacks
            const fixedData = {
                ...d,
                state: d.state || 'Unknown',
                title: d.title || `Standard ${d.id ? d.id.substring(0, 8) : index}`,
                color: d.color || '#007bff',
                subject: d.subject || 'Unknown Subject'
            };
            
            return fixedData;
        }).filter(d => d !== null);
        
        console.log(`After validation: ${validDataPoints.length}/${data.scatter_data.length} valid points`);
        
        if (validDataPoints.length === 0) {
            container.innerHTML = `
                <div class="alert alert-warning">
                    <h6><i class="fas fa-exclamation-triangle"></i> No Valid Data Points</h6>
                    <p>Found ${data.scatter_data.length} data points, but none have valid coordinates and properties.</p>
                    <small class="text-muted">Check the console for detailed validation errors.</small>
                </div>
            `;
            return;
        }
        
        console.log('First few valid data points:', validDataPoints.slice(0, 3));

        const margin = {top: 20, right: 80, bottom: 50, left: 50};
        const containerWidth = container.clientWidth || container.offsetWidth || 720;
        const containerHeight = container.clientHeight || container.offsetHeight || 480;
        const width = Math.max(containerWidth - margin.left - margin.right, 320);
        const height = Math.max(containerHeight - margin.top - margin.bottom, 320);

        d3.select(container).select('svg').remove();

        const svg = d3.select(container)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom);
        
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Create scales with validated data
        const xExtent = d3.extent(validDataPoints, d => d.x);
        const yExtent = d3.extent(validDataPoints, d => d.y);
        
        const xScale = d3.scaleLinear()
            .domain(xExtent)
            .range([0, width]);
        
        const yScale = d3.scaleLinear()
            .domain(yExtent)
            .range([height, 0]);
        
        // Add axes
        g.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(xScale));
        
        g.append('g')
            .call(d3.axisLeft(yScale));
        
        // Add axis labels
        g.append('text')
            .attr('transform', `translate(${width/2}, ${height + 40})`)
            .style('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('fill', '#666')
            .text('UMAP Dimension 1 (Semantic Similarity)');
        
        g.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', 0 - margin.left)
            .attr('x', 0 - (height / 2))
            .attr('dy', '1em')
            .style('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('fill', '#666')
            .text('UMAP Dimension 2 (Semantic Similarity)');
        
        // Add tooltip
        const tooltip = d3.select('body').append('div')
            .attr('class', 'tooltip')
            .style('position', 'absolute')
            .style('background', 'rgba(0,0,0,0.8)')
            .style('color', 'white')
            .style('padding', '8px')
            .style('border-radius', '4px')
            .style('pointer-events', 'none')
            .style('opacity', 0);
        
        // Plot points using simple, direct D3.js approach
        console.log('About to bind data to D3. Data length:', validDataPoints.length);
        
        // Clear any existing points completely
        g.selectAll('.point').remove();
        
        // Create points with additional validation to prevent undefined data issues
        try {
            const circles = g.selectAll('.point')
                .data(validDataPoints)
                .enter()
                .append('circle')
                .attr('class', 'point')
                .attr('cx', function(d, i) {
                    // Additional safety check to prevent undefined errors
                    if (!d || typeof d.x !== 'number' || isNaN(d.x)) {
                        console.warn(`Invalid data point at index ${i}:`, d);
                        return 0; // Default to 0 if invalid
                    }
                    return xScale(d.x);
                })
                .attr('cy', function(d, i) {
                    // Additional safety check to prevent undefined errors
                    if (!d || typeof d.y !== 'number' || isNaN(d.y)) {
                        console.warn(`Invalid data point at index ${i}:`, d);
                        return 0; // Default to 0 if invalid
                    }
                    return yScale(d.y);
                })
                .attr('r', 4)
                .attr('fill', function(d) {
                    if (!d) return '#007bff';
                    return d.color || '#007bff';
                })
                .attr('opacity', 0.7)
                .on('mouseover', function(event, d) {
                    if (d && d.title) {
                        tooltip.transition().duration(200).style('opacity', 0.9);
                        const header = d.manual_cluster_name || d.state || 'Standard';
                        const stateLine = d.manual_cluster_name && d.state ? `<div class="text-muted">${d.state}</div>` : '';
                        const subjectLine = d.subject ? `<div class="text-muted">Subject: ${d.subject}</div>` : '';
                        tooltip.html(`
                            <strong>${header}</strong>
                            ${stateLine}
                            <div>${d.title}</div>
                            ${subjectLine}
                        `.trim())
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 28) + 'px');
                    }
                })
                .on('mouseout', function() {
                    tooltip.transition().duration(500).style('opacity', 0);
                });
                
            console.log('Successfully created', circles.size(), 'circles');
        } catch (error) {
            console.error('Error creating circles:', error);
            container.innerHTML = `
                <div class="alert alert-danger">
                    <h6>D3.js Rendering Error</h6>
                    <p>${error.message}</p>
                </div>
            `;
            return;
        }
        
        // Clear any existing cluster elements
        g.selectAll('.cluster-boundary').remove();
        g.selectAll('.cluster-center').remove();
        g.selectAll('.cluster-label').remove();
        
        // Plot cluster boundaries and labels with validation
        console.log('Cluster rendering - data.clusters:', data.clusters);
        
        if (data.clusters && Array.isArray(data.clusters)) {
            console.log(`Found ${data.clusters.length} clusters in data`);
            
            // Debug each cluster
            data.clusters.forEach((cluster, i) => {
                console.log(`Cluster ${i}:`, {
                    id: cluster.id,
                    name: cluster.name,
                    center: cluster.center,
                    radius: cluster.radius,
                    standards_count: cluster.standards_count
                });
            });
            
            // Filter valid clusters
            const validClusters = data.clusters.filter(cluster => 
                cluster && 
                cluster.center && 
                typeof cluster.center.x === 'number' && 
                typeof cluster.center.y === 'number' &&
                !isNaN(cluster.center.x) && 
                !isNaN(cluster.center.y) &&
                typeof cluster.radius === 'number' &&
                !isNaN(cluster.radius) &&
                cluster.radius > 0
            );
            
            console.log(`${validClusters.length} valid clusters after filtering`);
            
            // Plot cluster boundary circles
            const boundaryCircles = g.selectAll('.cluster-boundary')
                .data(validClusters)
                .enter().append('circle')
                .attr('class', 'cluster-boundary')
                .attr('cx', d => {
                    const cx = xScale(d.center.x);
                    console.log(`Cluster ${d.id} cx:`, cx);
                    return cx;
                })
                .attr('cy', d => {
                    const cy = yScale(d.center.y);
                    console.log(`Cluster ${d.id} cy:`, cy);
                    return cy;
                })
                .attr('r', d => {
                    // Convert radius from data space to pixel space
                    const radiusInPixels = Math.abs(xScale(d.radius) - xScale(0));
                    const effectiveRadius = Math.max(radiusInPixels, 20); // Minimum radius of 20px
                    console.log(`Cluster ${d.id} radius: data=${d.radius}, pixels=${radiusInPixels}, effective=${effectiveRadius}`);
                    return effectiveRadius;
                })
                .attr('fill', 'none')
                .attr('stroke', '#666')
                .attr('stroke-width', 2)
                .attr('stroke-dasharray', '5,5')
                .attr('opacity', 0.6);
                
            console.log('Created boundary circles:', boundaryCircles.size());
            
            // Plot cluster center dots
            g.selectAll('.cluster-center')
                .data(validClusters)
                .enter().append('circle')
                .attr('class', 'cluster-center')
                .attr('cx', d => xScale(d.center.x))
                .attr('cy', d => yScale(d.center.y))
                .attr('r', 4)
                .attr('fill', '#333')
                .attr('stroke', '#fff')
                .attr('stroke-width', 2);
            
            // Add cluster topic labels with numbering
            g.selectAll('.cluster-label')
                .data(validClusters.filter(cluster => cluster.name))
                .enter().append('text')
                .attr('class', 'cluster-label')
                .attr('x', d => xScale(d.center.x))
                .attr('y', d => {
                    const radiusInPixels = Math.abs(xScale(d.radius) - xScale(0));
                    const effectiveRadius = Math.max(radiusInPixels, 20);
                    return yScale(d.center.y) - effectiveRadius - 10; // Position above the boundary circle
                })
                .attr('text-anchor', 'middle')
                .style('font-size', '14px')
                .style('font-weight', 'bold')
                .style('fill', '#333')
                .style('text-shadow', '1px 1px 2px rgba(255,255,255,0.8)')
                .text((d, i) => `${i + 1}. ${d.name}`);
        }
        
        // Update legend
        renderLegend(data.state_colors, data.manual_clusters);
        
        // Update cluster info
        renderClusterInfo(data.clusters);
        
        // Update info with clustering statistics
        const infoElement = document.getElementById('scatter-info');
        if (data.clustering_stats) {
            const stats = data.clustering_stats;
            infoElement.innerHTML = `
                <span class="text-muted">
                    <strong>${stats.total_standards}</strong> standards •
                    <strong>${stats.num_clusters}</strong> clusters •
                    <span class="text-success"><strong>${stats.clustered}</strong> clustered</span> •
                    <span class="text-warning"><strong>${stats.unclustered}</strong> unclustered</span> 
                    (<strong>${stats.clustering_rate}%</strong> clustered)
                </span>
            `;
        } else {
            // Fallback for older data format
            infoElement.textContent = `${data.total_standards} standards, ${data.clusters.length} clusters`;
        }
    }
    
    function renderLegend(stateColors, manualClusters) {
        const legend = document.getElementById('state-legend');
        if (!legend) {
            return;
        }
        legend.innerHTML = '';

        if (Array.isArray(manualClusters) && manualClusters.length) {
            manualClusters.forEach(cluster => {
                const item = document.createElement('div');
                item.className = 'state-legend-item';
                item.innerHTML = `
                    <div class="state-color" style="background-color: ${cluster.color}"></div>
                    <span>${cluster.name}</span>
                    <span class="text-muted small ms-2">${cluster.standards_count || 0}</span>
                `;
                legend.appendChild(item);
            });
            return;
        }

        Object.entries(stateColors || {}).forEach(([state, color]) => {
            const item = document.createElement('div');
            item.className = 'state-legend-item';
            item.innerHTML = `
                <div class="state-color" style="background-color: ${color}"></div>
                <span>${state}</span>
            `;
            legend.appendChild(item);
        });
    }
    
    // 3D scatter plot using Plotly.js
    function renderScatterPlot3D(data) {
        const container = document.getElementById('scatter-plot');
        
        console.log('renderScatterPlot3D called with data:', data);

        if (!data.scatter_data || !Array.isArray(data.scatter_data)) {
            container.innerHTML = '<div class="alert alert-warning">Invalid data format: missing scatter_data array.</div>';
            return;
        }

        renderLegend(data.state_colors, data.manual_clusters);
        
        if (data.scatter_data.length === 0) {
            container.innerHTML = `
                <div class="alert alert-info">
                    <h6><i class="fas fa-info-circle"></i> No Standards Found</h6>
                    <p>No standards with embeddings match your current filter criteria.</p>
                    <small class="text-muted">Try selecting different filters or check if embeddings have been generated.</small>
                </div>
            `;
            return;
        }
        
        // Validate and prepare 3D data
        const validDataPoints = data.scatter_data.filter(d => 
            d && typeof d.x === 'number' && typeof d.y === 'number' && 
            typeof d.z === 'number' && !isNaN(d.x) && !isNaN(d.y) && !isNaN(d.z)
        );
        
        if (validDataPoints.length === 0) {
            container.innerHTML = `
                <div class="alert alert-warning">
                    <h6><i class="fas fa-exclamation-triangle"></i> No Valid 3D Data</h6>
                    <p>No data points contain valid z-coordinates for 3D visualization.</p>
                    <small class="text-muted">Try switching to 2D mode or check the data generation.</small>
                </div>
            `;
            return;
        }
        
        console.log(`Creating 3D visualization with ${validDataPoints.length} valid points`);
        
        const useManualClusters = Array.isArray(data.manual_clusters) && data.manual_clusters.length > 0;
        const groups = {};
        const groupLabels = {};
        validDataPoints.forEach(point => {
            let key;
            if (useManualClusters) {
                key = point.manual_cluster_id || 'unassigned';
                groupLabels[key] = point.manual_cluster_name || 'Unassigned';
            } else {
                key = point.state || 'Unknown';
            }
            if (!groups[key]) {
                groups[key] = [];
            }
            groups[key].push(point);
        });

        const traces = Object.entries(groups).map(([key, points]) => ({
            x: points.map(p => p.x),
            y: points.map(p => p.y),
            z: points.map(p => p.z),
            mode: 'markers',
            type: 'scatter3d',
            name: useManualClusters ? groupLabels[key] : key,
            text: points.map(p => {
                const clusterLine = useManualClusters && p.manual_cluster_name ? `Cluster: ${p.manual_cluster_name}<br>` : '';
                const stateLine = p.state ? `State: ${p.state}<br>` : '';
                return `${p.title}<br>${clusterLine}${stateLine}Subject: ${p.subject || 'Unknown'}`;
            }),
            hovertemplate: '%{text}<extra></extra>',
            marker: {
                size: 4,
                color: points[0].color || '#007bff',
                opacity: 0.7
            }
        }));
        
        // Add 3D cluster boundaries if cluster data is available
        if (data.clusters && Array.isArray(data.clusters)) {
            console.log('Adding 3D cluster boundaries for', data.clusters.length, 'clusters');
            
            data.clusters.forEach((cluster, index) => {
                // Validate cluster has 3D center and radius
                if (!cluster.center || typeof cluster.center.z !== 'number' || !cluster.radius_3d) {
                    console.warn(`Cluster ${cluster.id} missing 3D data, skipping boundary`);
                    return;
                }
                
                // Get points belonging to this cluster
                const clusterPoints = validDataPoints.filter(point => point.cluster === cluster.id);
                
                if (clusterPoints.length < 3) {
                    console.log(`Cluster ${cluster.id} has too few points for boundary (${clusterPoints.length})`);
                    return;
                }
                
                console.log(`Creating boundary for cluster ${cluster.id} with ${clusterPoints.length} points`);
                
                // Create mesh3d trace for cluster boundary using cluster points
                const clusterTrace = {
                    type: 'mesh3d',
                    x: clusterPoints.map(p => p.x),
                    y: clusterPoints.map(p => p.y),
                    z: clusterPoints.map(p => p.z),
                    alphahull: 2.0,  // Controls boundary tightness (higher = looser)
                    opacity: 0.15,   // Semi-transparent
                    color: clusterPoints[0]?.color || `hsl(${(index * 60) % 360}, 70%, 60%)`,
                    name: `${cluster.name} Boundary`,
                    showlegend: false,  // Don't clutter legend
                    hoverinfo: 'text',
                    text: `${index + 1}. ${cluster.name}<br>Size: ${cluster.standards_count} standards<br>States: ${cluster.states.join(', ')}`,
                    lighting: {
                        ambient: 0.4,
                        diffuse: 0.6,
                        specular: 0.1
                    }
                };
                
                traces.push(clusterTrace);
            });
        }
        
        // Configure 3D layout
        const layout = {
            title: {
                text: '3D Educational Standards Clustering',
                font: { size: 16 }
            },
            scene: {
                xaxis: {
                    title: 'UMAP Dimension 1 (Semantic Similarity)',
                    titlefont: { size: 12 },
                    tickfont: { size: 10 }
                },
                yaxis: {
                    title: 'UMAP Dimension 2 (Semantic Similarity)', 
                    titlefont: { size: 12 },
                    tickfont: { size: 10 }
                },
                zaxis: {
                    title: 'UMAP Dimension 3 (Educational Context)',
                    titlefont: { size: 12 },
                    tickfont: { size: 10 }
                },
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.5 },
                    center: { x: 0, y: 0, z: 0 },
                    up: { x: 0, y: 0, z: 1 }
                }
            },
            legend: {
                orientation: 'v',
                x: 1.02,
                y: 1,
                font: { size: 10 }
            },
            margin: { l: 0, r: 0, t: 40, b: 0 },
            hovermode: 'closest'
        };
        
        // Configure Plotly options
        const config = {
            displayModeBar: true,
            modeBarButtonsToAdd: [{
                name: 'Reset Camera',
                icon: Plotly.Icons.home,
                click: function(gd) {
                    Plotly.relayout(gd, {
                        'scene.camera.eye': { x: 1.5, y: 1.5, z: 1.5 },
                        'scene.camera.center': { x: 0, y: 0, z: 0 },
                        'scene.camera.up': { x: 0, y: 0, z: 1 }
                    });
                }
            }],
            responsive: true
        };
        
        // Create the 3D plot
        Plotly.newPlot(container, traces, layout, config).then(() => {
            console.log('3D scatter plot created successfully');
            
            // Add navigation and controls
            const controlsDiv = document.createElement('div');
            controlsDiv.className = 'mt-2';
            controlsDiv.innerHTML = `
                <div class="row align-items-center">
                    <div class="col-md-8">
                        <div class="alert alert-info py-2 mb-0">
                            <small>
                                <strong>3D Navigation:</strong> 
                                <i class="fas fa-mouse-pointer"></i> Click and drag to rotate • 
                                <i class="fas fa-search-plus"></i> Scroll to zoom • 
                                <i class="fas fa-arrows-alt"></i> Shift+drag to pan • 
                                <i class="fas fa-home"></i> Use toolbar to reset view
                            </small>
                        </div>
                    </div>
                    <div class="col-md-4 text-end">
                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" id="toggle3dBoundaries" checked>
                            <label class="form-check-label small" for="toggle3dBoundaries">
                                <i class="fas fa-circle-notch"></i> Cluster Boundaries
                            </label>
                        </div>
                    </div>
                </div>
            `;
            container.appendChild(controlsDiv);
            
            // Add toggle functionality for cluster boundaries
            const boundaryToggle = document.getElementById('toggle3dBoundaries');
            if (boundaryToggle) {
                boundaryToggle.addEventListener('change', function() {
                    const shouldShow = this.checked;
                    console.log('Toggling cluster boundaries:', shouldShow);
                    
                    // Find boundary traces (mesh3d traces)
                    const update = {};
                    traces.forEach((trace, index) => {
                        if (trace.type === 'mesh3d') {
                            update[`visible[${index}]`] = shouldShow;
                        }
                    });
                    
                    if (Object.keys(update).length > 0) {
                        Plotly.restyle(container, update);
                    }
                });
            }
        }).catch(error => {
            console.error('Error creating 3D plot:', error);
            container.innerHTML = `
                <div class="alert alert-danger">
                    <h6>3D Visualization Error</h6>
                    <p>Failed to create 3D visualization: ${error.message}</p>
                    <small class="text-muted">Try switching to 2D mode or refresh the page.</small>
                </div>
            `;
        });
    }
    
    function renderClusterInfo(clusters) {
        const container = document.getElementById('cluster-info');
        container.innerHTML = '';
        
        // Debug logging
        console.log('renderClusterInfo called with clusters:', clusters);
        if (clusters && clusters.length > 0) {
            console.log('First cluster structure:', clusters[0]);
        }
        
        if (!clusters || clusters.length === 0) {
            container.innerHTML = `
                <div class="alert alert-info">
                    <h6><i class="fas fa-info-circle"></i> No Clusters Found</h6>
                    <p class="mb-0">No distinct clusters were identified in the current dataset. Try adjusting the clustering parameters or including more data.</p>
                </div>
            `;
            return;
        }
        
        // Add Understanding Your Results section only (removed redundant cluster info)
        const resultsHelp = document.createElement('div');
        resultsHelp.className = 'alert alert-light border-start border-primary border-3 mb-4';
        resultsHelp.innerHTML = `
            <h6><i class="fas fa-chart-pie text-primary"></i> Understanding Your Results</h6>
            <div class="row">
                <div class="col-12">
                    <strong>What to look for:</strong>
                    <ul class="small mb-0">
                        <li><strong>Large clusters:</strong> Concepts taught across many states</li>
                        <li><strong>State-specific clusters:</strong> Unique state approaches</li>
                        <li><strong>Grade-level patterns:</strong> How concepts progress through grades</li>
                    </ul>
                </div>
            </div>
        `;
        
        container.appendChild(resultsHelp);
        
        clusters.forEach((cluster, index) => {
            const clusterId = `cluster-${cluster.id}`;
            const collapseId = `collapse-${cluster.id}`;
            
            const div = document.createElement('div');
            div.className = 'card mb-3';
            div.innerHTML = `
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start">
                        <h6 class="card-title mb-2">
                            <span class="badge bg-primary me-2">${index + 1}</span>
                            ${cluster.name}
                        </h6>
                        <span class="badge bg-secondary">${cluster.standards_count} standards</span>
                    </div>
                    <div class="mb-3">
                        <strong>Key Topics:</strong><br>
                        <div class="mt-2">
                            ${(cluster.key_topics && cluster.key_topics.length > 0) 
                                ? cluster.key_topics.map((topic, index) => {
                                    // Determine badge color based on topic content for better visual organization
                                    let badgeClass = 'bg-primary';
                                    const topicLower = topic.toLowerCase();
                                    
                                    if (topicLower.includes('government') || topicLower.includes('constitution') || 
                                        topicLower.includes('democracy') || topicLower.includes('citizen')) {
                                        badgeClass = 'bg-info';
                                    } else if (topicLower.includes('geography') || topicLower.includes('map') || 
                                              topicLower.includes('location') || topicLower.includes('region')) {
                                        badgeClass = 'bg-success';
                                    } else if (topicLower.includes('history') || topicLower.includes('historical') || 
                                              topicLower.includes('period') || topicLower.includes('era')) {
                                        badgeClass = 'bg-warning';
                                    } else if (topicLower.includes('math') || topicLower.includes('number') || 
                                              topicLower.includes('algebra') || topicLower.includes('geometry')) {
                                        badgeClass = 'bg-danger';
                                    } else if (topicLower.includes('science') || topicLower.includes('experiment') || 
                                              topicLower.includes('energy') || topicLower.includes('matter')) {
                                        badgeClass = 'bg-dark';
                                    }
                                    
                                    return `<span class="badge ${badgeClass} me-1 mb-1" title="Topic ${index + 1}">${topic}</span>`;
                                }).join('')
                                : '<span class="badge bg-secondary">No topics identified</span>'
                            }
                        </div>
                        ${cluster.key_topics && cluster.key_topics.length > 8 
                            ? `<small class="text-muted mt-1 d-block">${cluster.key_topics.length} key topics identified</small>`
                            : ''
                        }
                    </div>
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <small class="text-muted"><strong>Leading States:</strong></small><br>
                            <small>${(cluster.states && cluster.states.length > 0) 
                                ? cluster.states.slice(0, 5).join(', ') + (cluster.states.length > 5 ? '...' : '')
                                : 'No states identified'
                            }</small>
                        </div>
                        <div class="col-md-6">
                            <small class="text-muted"><strong>State Coverage:</strong></small><br>
                            <small>${cluster.states ? cluster.states.length : 0} states represented</small>
                        </div>
                    </div>
                    
                    <!-- Show Standards Button -->
                    <button class="btn btn-outline-primary btn-sm" type="button" 
                            data-bs-toggle="collapse" data-bs-target="#${collapseId}" 
                            aria-expanded="false" aria-controls="${collapseId}"
                            onclick="toggleStandardsView('${collapseId}', this)">
                        <i class="fas fa-eye"></i> Show Standards
                    </button>
                    
                    <!-- Collapsible Standards List -->
                    <div class="collapse mt-3" id="${collapseId}">
                        <div class="border rounded p-3 bg-light">
                            <h6 class="mb-3">
                                <i class="fas fa-list"></i> 
                                Standards in this cluster (${cluster.standards_count})
                            </h6>
                            <div class="standards-list" id="standards-${cluster.id}">
                                ${renderStandardsList(cluster.standards)}
                            </div>
                        </div>
                    </div>
                </div>
            `;
            container.appendChild(div);
        });
    }
    
    // Helper function to render standards list
    function renderStandardsList(standards) {
        if (!standards || standards.length === 0) {
            return '<div class="alert alert-warning">No standards data available.</div>';
        }
        
        return standards.map(standard => {
            const gradesBadges = standard.grade_levels.map(grade => 
                `<span class="badge bg-info me-1">Grade ${grade}</span>`
            ).join('');
            
            return `
                <div class="card mb-2">
                    <div class="card-body py-2">
                        <div class="d-flex justify-content-between align-items-start">
                            <div class="flex-grow-1">
                                <h6 class="card-title mb-1">${standard.title}</h6>
                                ${standard.description ? `<p class="card-text small text-muted mb-2">${standard.description}</p>` : ''}
                            </div>
                            <div class="text-end ms-3">
                                <span class="badge bg-primary">${standard.state}</span>
                            </div>
                        </div>
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                ${gradesBadges}
                                ${standard.subject_area ? `<span class="badge bg-secondary">${standard.subject_area}</span>` : ''}
                            </div>
                            <small class="text-muted">${standard.state_name}</small>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }
    
    // Toggle standards view function
    function toggleStandardsView(collapseId, button) {
        const collapseElement = document.getElementById(collapseId);
        const clusterCard = button.closest('.card');
        
        // Listen for Bootstrap collapse events to update button text
        collapseElement.addEventListener('shown.bs.collapse', function () {
            button.innerHTML = '<i class="fas fa-eye-slash"></i> Hide Standards';
            clusterCard.classList.add('expanded');
        });
        
        collapseElement.addEventListener('hidden.bs.collapse', function () {
            button.innerHTML = '<i class="fas fa-eye"></i> Show Standards';
            clusterCard.classList.remove('expanded');
        });
    }
    
    // Heatmap visualization
    function renderHeatmap(data) {
        const container = document.getElementById('heatmap');
        container.innerHTML = '';
        
        const margin = {top: 50, right: 50, bottom: 50, left: 50};
        const cellSize = Math.min(
            (container.clientWidth - margin.left - margin.right) / data.states.length,
            (container.clientHeight - margin.top - margin.bottom) / data.states.length
        );
        
        const width = cellSize * data.states.length;
        const height = cellSize * data.states.length;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom);
        
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Color scale
        const colorScale = d3.scaleSequential(d3.interpolateBlues)
            .domain([0, 1]);
        
        // Create tooltip
        const tooltip = d3.select('body').append('div')
            .attr('class', 'heatmap-tooltip')
            .style('position', 'absolute')
            .style('background', 'rgba(0,0,0,0.8)')
            .style('color', 'white')
            .style('padding', '8px')
            .style('border-radius', '4px')
            .style('pointer-events', 'none')
            .style('opacity', 0);
        
        // Create grid
        data.similarity_matrix.forEach((row, i) => {
            row.forEach((value, j) => {
                g.append('rect')
                    .attr('x', j * cellSize)
                    .attr('y', i * cellSize)
                    .attr('width', cellSize)
                    .attr('height', cellSize)
                    .attr('fill', colorScale(value))
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 1)
                    .on('mouseover', function(event) {
                        tooltip.transition().duration(200).style('opacity', .9);
                        tooltip.html(`${data.states[i]} ↔ ${data.states[j]}<br/>Similarity: ${value.toFixed(3)}`)
                            .style('left', (event.pageX + 10) + 'px')
                            .style('top', (event.pageY - 28) + 'px');
                    })
                    .on('mouseout', function() {
                        tooltip.transition().duration(500).style('opacity', 0);
                    });
            });
        });
        
        // Add state labels
        g.selectAll('.state-label-x')
            .data(data.states)
            .enter().append('text')
            .attr('class', 'state-label-x')
            .attr('x', (d, i) => (i + 0.5) * cellSize)
            .attr('y', -10)
            .attr('text-anchor', 'middle')
            .style('font-size', '10px')
            .text(d => d);
        
        g.selectAll('.state-label-y')
            .data(data.states)
            .enter().append('text')
            .attr('class', 'state-label-y')
            .attr('x', -10)
            .attr('y', (d, i) => (i + 0.5) * cellSize + 4)
            .attr('text-anchor', 'end')
            .style('font-size', '10px')
            .text(d => d);
        
        document.getElementById('heatmap-info').textContent = 
            `${data.states.length} states compared`;
    }
    
    // Enhanced heat map rendering with multiple types support
    function renderEnhancedHeatmap(data, heatmapType = 'state') {
        const container = document.getElementById('heatmap');
        container.innerHTML = '';
        
        if (data.error) {
            showHeatMapError(data.error);
            return;
        }
        
        // Handle different data formats
        let labels, matrix;
        
        if (heatmapType === 'cluster') {
            labels = data.cluster_names || [];
            matrix = data.similarity_matrix || [];
        } else if (data.row_labels && data.col_labels) {
            // Enhanced matrix format
            labels = { row: data.row_labels, col: data.col_labels };
            matrix = data.matrix || [];
        } else {
            // Traditional state matrix format
            labels = data.states || [];
            matrix = data.similarity_matrix || [];
        }
        
        if (!matrix.length) {
            showHeatMapError('No data available for visualization');
            return;
        }
        
        // Calculate dimensions
        const isSquareMatrix = !labels.row; // Square matrix if labels aren't separate
        const numRows = isSquareMatrix ? labels.length : labels.row.length;
        const numCols = isSquareMatrix ? labels.length : labels.col.length;
        
        const margin = {top: 80, right: 80, bottom: 80, left: 80};
        const cellSize = Math.min(
            (container.clientWidth - margin.left - margin.right) / numCols,
            (container.clientHeight - margin.top - margin.bottom) / numRows,
            40 // Maximum cell size
        );
        
        const width = cellSize * numCols;
        const height = cellSize * numRows;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom);
        
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Get color scheme
        const colorScheme = document.getElementById('heatmap-colorscheme')?.value || 'blues';
        const colorScale = getColorScale(colorScheme, matrix, heatmapType);
        
        // Create enhanced tooltip
        const tooltip = createEnhancedTooltip();
        
        // Create grid
        matrix.forEach((row, i) => {
            row.forEach((value, j) => {
                const rect = g.append('rect')
                    .attr('x', j * cellSize)
                    .attr('y', i * cellSize)
                    .attr('width', cellSize)
                    .attr('height', cellSize)
                    .attr('fill', colorScale(value))
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 1)
                    .style('cursor', 'pointer')
                    .on('mouseover', function(event) {
                        // Enhanced tooltip content based on heat map type
                        const tooltipContent = generateTooltipContent(data, i, j, value, heatmapType, isSquareMatrix ? labels : labels.row, isSquareMatrix ? labels : labels.col);
                        
                        tooltip.transition().duration(200).style('opacity', 0.95);
                        tooltip.html(tooltipContent)
                            .style('left', (event.pageX + 10) + 'px')
                            .style('top', (event.pageY - 10) + 'px');
                            
                        // Highlight row and column
                        highlightRowCol(g, i, j, numRows, numCols, cellSize);
                    })
                    .on('mouseout', function() {
                        tooltip.transition().duration(500).style('opacity', 0);
                        removeHighlights(g);
                    })
                    .on('click', function() {
                        // Interactive feature: clicking links to other visualizations
                        handleHeatMapCellClick(data, i, j, heatmapType, isSquareMatrix ? labels : labels.row, isSquareMatrix ? labels : labels.col);
                    });
            });
        });
        
        // Add labels with rotation for better readability
        if (isSquareMatrix) {
            addSquareMatrixLabels(g, labels, cellSize, numRows);
        } else {
            addAsymmetricMatrixLabels(g, labels.row, labels.col, cellSize, numRows, numCols);
        }
        
        // Add color legend
        addColorLegend(svg, colorScale, heatmapType, width + margin.left + 10, margin.top);
        
        // Update info display
        const infoText = generateInfoText(data, heatmapType, numRows, numCols);
        document.getElementById('heatmap-info').innerHTML = infoText;
    }
    
    // Simplified function for topic coverage heat map
    function renderTopicCoverageHeatmap(data) {
        const container = document.getElementById('heatmap');
        container.innerHTML = '';
        
        if (data.error) {
            showHeatMapError(data.error);
            return;
        }
        
        // Extract topic coverage data
        const topicNames = data.topic_names || [];
        const stateCodes = data.state_codes || [];
        const coverageMatrix = data.coverage_matrix || [];
        
        if (topicNames.length === 0 || stateCodes.length === 0) {
            showHeatMapError('No topic coverage data available');
            return;
        }
        
        const numRows = topicNames.length; // Topics (rows)
        const numCols = stateCodes.length; // States (columns)
        
        // Conservative sizing to prevent overflow - increased left margin for topic names
        const margin = { top: 80, right: 100, bottom: 50, left: 250 };
        const maxAvailableWidth = 900; // Conservative max width
        const maxAvailableHeight = 600; // Conservative max height
        
        const maxCellSize = Math.min(
            (maxAvailableWidth - margin.left - margin.right) / numCols,
            (maxAvailableHeight - margin.top - margin.bottom) / numRows,
            30 // Smaller max cell size
        );
        const cellSize = Math.max(maxCellSize, 12); // Smaller minimum size
        
        const width = Math.min(numCols * cellSize, maxAvailableWidth - margin.left - margin.right);
        const height = Math.min(numRows * cellSize, maxAvailableHeight - margin.top - margin.bottom);
        
        // Create SVG
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom);
        
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Get color scale
        const colorScheme = document.getElementById('heatmap-colorscheme')?.value || 'blues';
        const colorScale = getTopicCoverageColorScale(colorScheme, coverageMatrix);
        
        // Create enhanced tooltip
        const tooltip = createEnhancedTooltip();
        
        // Draw heat map cells
        coverageMatrix.forEach((row, i) => {
            row.forEach((value, j) => {
                g.append('rect')
                    .attr('x', j * cellSize)
                    .attr('y', i * cellSize)
                    .attr('width', cellSize)
                    .attr('height', cellSize)
                    .attr('fill', colorScale(value))
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 1)
                    .style('cursor', 'pointer')
                    .on('mouseover', function(event) {
                        const tooltipContent = generateTopicCoverageTooltip(data, i, j, value, topicNames[i], stateCodes[j]);
                        
                        tooltip.transition().duration(200).style('opacity', 0.95);
                        tooltip.html(tooltipContent)
                            .style('left', (event.pageX + 10) + 'px')
                            .style('top', (event.pageY - 28) + 'px');
                        
                        // Highlight row and column
                        g.selectAll('rect').style('opacity', 0.6);
                        g.selectAll('rect').filter((d, idx) => Math.floor(idx / numCols) === i || idx % numCols === j).style('opacity', 1);
                    })
                    .on('mouseout', function() {
                        tooltip.transition().duration(500).style('opacity', 0);
                        g.selectAll('rect').style('opacity', 1);
                    })
                    .on('click', function() {
                        handleTopicCoverageCellClick(data, i, j, topicNames[i], stateCodes[j]);
                    });
            });
        });
        
        // Add row labels (topics)
        g.selectAll('.row-label')
            .data(topicNames)
            .enter()
            .append('text')
            .attr('class', 'row-label')
            .attr('x', -10)
            .attr('y', (d, i) => i * cellSize + cellSize / 2)
            .attr('dy', '0.35em')
            .style('text-anchor', 'end')
            .style('font-size', '12px')
            .style('font-weight', '500')
            .text(d => d.length > 40 ? d.substring(0, 40) + '...' : d)
            .append('title')
            .text(d => d);
        
        // Add column labels (states)
        g.selectAll('.col-label')
            .data(stateCodes)
            .enter()
            .append('text')
            .attr('class', 'col-label')
            .attr('x', (d, i) => i * cellSize + cellSize / 2)
            .attr('y', -10)
            .style('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('font-weight', '500')
            .text(d => d);
        
        // Add color legend
        addTopicCoverageLegend(svg, colorScale, width + margin.left + 10, margin.top);
        
        // Update info display
        document.getElementById('heatmap-info').innerHTML = `${numRows} topics × ${numCols} states • Click cells for details`;
    }
    
    function getTopicCoverageColorScale(colorScheme, matrix) {
        const flatValues = matrix.flat();
        const minVal = Math.min(...flatValues);
        const maxVal = Math.max(...flatValues);
        
        let colorInterpolator;
        switch(colorScheme) {
            case 'reds': colorInterpolator = d3.interpolateReds; break;
            case 'viridis': colorInterpolator = d3.interpolateViridis; break;
            case 'plasma': colorInterpolator = d3.interpolatePlasma; break;
            case 'rdbu': colorInterpolator = d3.interpolateRdBu; break;
            default: colorInterpolator = d3.interpolateBlues;
        }
        
        return d3.scaleSequential(colorInterpolator)
            .domain([minVal, maxVal]);
    }
    
    function generateTopicCoverageTooltip(data, i, j, value, topicName, stateCode) {
        const metadata = data.topic_metadata && data.topic_metadata[i];
        const percentage = Math.round(value * 100);
        
        let content = `
            <div class="tooltip-header">
                <strong>${topicName}</strong> in <strong>${stateCode}</strong>
            </div>
            <div class="tooltip-body">
                <div class="tooltip-metric">
                    <span class="metric-label">Coverage:</span>
                    <span class="metric-value">${percentage}%</span>
                </div>
        `;
        
        if (metadata) {
            content += `
                <div class="tooltip-metric">
                    <span class="metric-label">Total Standards:</span>
                    <span class="metric-value">${metadata.total_standards}</span>
                </div>
                <div class="tooltip-metric">
                    <span class="metric-label">States Covered:</span>
                    <span class="metric-value">${metadata.states_covered}</span>
                </div>
            `;
            
            if (metadata.sample_standards && metadata.sample_standards.length > 0) {
                content += `
                    <div class="tooltip-section">
                        <div class="tooltip-label">Sample Standards:</div>
                        <ul class="tooltip-list">
                            ${metadata.sample_standards.slice(0, 3).map(std => `<li>${std}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
        }
        
        content += `</div>`;
        return content;
    }
    
    function addTopicCoverageLegend(svg, colorScale, x, y) {
        const legendHeight = 150;
        const legendWidth = 20;
        
        const legend = svg.append('g')
            .attr('class', 'legend')
            .attr('transform', `translate(${x},${y})`);
        
        // Create gradient
        const defs = svg.append('defs');
        const gradient = defs.append('linearGradient')
            .attr('id', 'topic-coverage-legend-gradient')
            .attr('x1', '0%').attr('y1', '0%')
            .attr('x2', '0%').attr('y2', '100%');
        
        const numStops = 10;
        for (let i = 0; i <= numStops; i++) {
            const t = i / numStops;
            gradient.append('stop')
                .attr('offset', `${t * 100}%`)
                .attr('stop-color', colorScale(colorScale.domain()[1] * (1 - t) + colorScale.domain()[0] * t));
        }
        
        // Legend rectangle
        legend.append('rect')
            .attr('width', legendWidth)
            .attr('height', legendHeight)
            .style('fill', 'url(#topic-coverage-legend-gradient)');
        
        // Legend scale
        const legendScale = d3.scaleLinear()
            .domain(colorScale.domain())
            .range([legendHeight, 0]);
        
        const legendAxis = d3.axisRight(legendScale)
            .ticks(5)
            .tickFormat(d => Math.round(d * 100) + '%');
        
        legend.append('g')
            .attr('transform', `translate(${legendWidth}, 0)`)
            .call(legendAxis);
        
        // Legend title
        legend.append('text')
            .attr('x', legendWidth / 2)
            .attr('y', -10)
            .style('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('font-weight', 'bold')
            .text('Coverage');
    }
    
    function handleTopicCoverageCellClick(data, i, j, topicName, stateCode) {
        console.log(`Topic coverage cell clicked: ${topicName} in ${stateCode}`);
        // Could add interaction with scatter plot or other visualizations here
    }
    
    function updateTopicCoverageStats(data) {
        const statsElement = document.getElementById('heatmap-stats');
        
        if (!data || data.error) {
            statsElement.innerHTML = '<span class="text-muted">No data available</span>';
            return;
        }
        
        const numTopics = data.total_topics || 0;
        const numStates = data.total_states || 0;
        const numStandards = data.total_standards || 0;
        
        statsElement.innerHTML = `📊 ${numTopics} topics • ${numStates} states • ${numStandards} standards analyzed`;
    }
    
    // Enhanced State Analysis rendering function
    function renderEnhancedStateHeatmap(data) {
        const container = document.getElementById('heatmap');
        container.innerHTML = '';
        
        if (data.error) {
            showHeatMapError(data.error);
            return;
        }
        
        // Extract enhanced state data
        const rowLabels = data.row_labels || [];
        const colLabels = data.col_labels || [];
        const matrix = data.matrix || [];
        
        if (rowLabels.length === 0 || colLabels.length === 0) {
            showHeatMapError('No enhanced state data available');
            return;
        }
        
        const numRows = rowLabels.length;
        const numCols = colLabels.length;
        
        // Conservative sizing to prevent overflow
        const margin = { top: 60, right: 100, bottom: 60, left: 60 };
        const maxAvailableWidth = 800; // Conservative max width
        const maxAvailableHeight = 600; // Conservative max height
        
        const maxCellSize = Math.min(
            (maxAvailableWidth - margin.left - margin.right) / numCols,
            (maxAvailableHeight - margin.top - margin.bottom) / numRows,
            35 // Reasonable max cell size
        );
        const cellSize = Math.max(maxCellSize, 15); // Minimum readable size
        
        const width = Math.min(numCols * cellSize, maxAvailableWidth - margin.left - margin.right);
        const height = Math.min(numRows * cellSize, maxAvailableHeight - margin.top - margin.bottom);
        
        // Create SVG
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom);
        
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Get color scale
        const colorScheme = document.getElementById('heatmap-colorscheme')?.value || 'blues';
        const colorScale = getEnhancedStateColorScale(colorScheme, matrix);
        
        // Create enhanced tooltip
        const tooltip = createEnhancedTooltip();
        
        // Draw heat map cells
        matrix.forEach((row, i) => {
            row.forEach((value, j) => {
                g.append('rect')
                    .attr('x', j * cellSize)
                    .attr('y', i * cellSize)
                    .attr('width', cellSize)
                    .attr('height', cellSize)
                    .attr('fill', colorScale(value))
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 1)
                    .style('cursor', 'pointer')
                    .on('mouseover', function(event) {
                        const tooltipContent = generateEnhancedStateTooltip(data, i, j, value, rowLabels[i], colLabels[j]);
                        
                        tooltip.transition().duration(200).style('opacity', 0.95);
                        tooltip.html(tooltipContent)
                            .style('left', (event.pageX + 10) + 'px')
                            .style('top', (event.pageY - 28) + 'px');
                        
                        // Highlight row and column
                        g.selectAll('rect').style('opacity', 0.6);
                        g.selectAll('rect').filter((d, idx) => Math.floor(idx / numCols) === i || idx % numCols === j).style('opacity', 1);
                    })
                    .on('mouseout', function() {
                        tooltip.transition().duration(500).style('opacity', 0);
                        g.selectAll('rect').style('opacity', 1);
                    })
                    .on('click', function() {
                        handleEnhancedStateCellClick(data, i, j, rowLabels[i], colLabels[j]);
                    });
            });
        });
        
        // Add row labels (states)
        g.selectAll('.row-label')
            .data(rowLabels)
            .enter()
            .append('text')
            .attr('class', 'row-label')
            .attr('x', -10)
            .attr('y', (d, i) => i * cellSize + cellSize / 2)
            .attr('dy', '0.35em')
            .style('text-anchor', 'end')
            .style('font-size', '12px')
            .style('font-weight', '500')
            .text(d => d);
        
        // Add column labels (states)
        g.selectAll('.col-label')
            .data(colLabels)
            .enter()
            .append('text')
            .attr('class', 'col-label')
            .attr('x', (d, i) => i * cellSize + cellSize / 2)
            .attr('y', -10)
            .style('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('font-weight', '500')
            .text(d => d);
        
        // Add color legend
        addEnhancedStateLegend(svg, colorScale, width + margin.left + 10, margin.top);
        
        // Update info display
        document.getElementById('heatmap-info').innerHTML = `${numRows} × ${numCols} states • Click cells for comparison details`;
    }
    
    function getEnhancedStateColorScale(colorScheme, matrix) {
        const flatValues = matrix.flat();
        const minVal = Math.min(...flatValues);
        const maxVal = Math.max(...flatValues);
        
        let colorInterpolator;
        switch(colorScheme) {
            case 'reds': colorInterpolator = d3.interpolateReds; break;
            case 'viridis': colorInterpolator = d3.interpolateViridis; break;
            case 'plasma': colorInterpolator = d3.interpolatePlasma; break;
            case 'rdbu': colorInterpolator = d3.interpolateRdBu; break;
            default: colorInterpolator = d3.interpolateBlues;
        }
        
        return d3.scaleSequential(colorInterpolator)
            .domain([minVal, maxVal]);
    }
    
    function generateEnhancedStateTooltip(data, i, j, value, stateA, stateB) {
        const percentage = Math.round(value * 100);
        const similarity = value >= 0.8 ? 'Very Similar' : 
                          value >= 0.6 ? 'Similar' : 
                          value >= 0.4 ? 'Moderately Similar' : 'Different';
        
        let content = `
            <div class="tooltip-header">
                <strong>${stateA}</strong> vs <strong>${stateB}</strong>
            </div>
            <div class="tooltip-body">
                <div class="tooltip-metric">
                    <span class="metric-label">Similarity:</span>
                    <span class="metric-value">${percentage}%</span>
                </div>
                <div class="tooltip-metric">
                    <span class="metric-label">Assessment:</span>
                    <span class="metric-value">${similarity}</span>
                </div>
        `;
        
        if (i !== j) {
            content += `
                <div class="tooltip-section">
                    <div class="tooltip-label">Analysis:</div>
                    <p class="small">
                        ${stateA} and ${stateB} have ${similarity.toLowerCase()} educational standards based on 
                        semantic analysis of their content embeddings.
                    </p>
                </div>
            `;
        } else {
            content += `
                <div class="tooltip-section">
                    <div class="tooltip-label">Self-comparison:</div>
                    <p class="small">Perfect similarity (100%) - same state</p>
                </div>
            `;
        }
        
        content += `</div>`;
        return content;
    }
    
    function addEnhancedStateLegend(svg, colorScale, x, y) {
        const legendHeight = 150;
        const legendWidth = 20;
        
        const legend = svg.append('g')
            .attr('class', 'legend')
            .attr('transform', `translate(${x},${y})`);
        
        // Create gradient
        const defs = svg.append('defs');
        const gradient = defs.append('linearGradient')
            .attr('id', 'enhanced-state-legend-gradient')
            .attr('x1', '0%').attr('y1', '0%')
            .attr('x2', '0%').attr('y2', '100%');
        
        const numStops = 10;
        for (let i = 0; i <= numStops; i++) {
            const t = i / numStops;
            gradient.append('stop')
                .attr('offset', `${t * 100}%`)
                .attr('stop-color', colorScale(colorScale.domain()[1] * (1 - t) + colorScale.domain()[0] * t));
        }
        
        // Legend rectangle
        legend.append('rect')
            .attr('width', legendWidth)
            .attr('height', legendHeight)
            .style('fill', 'url(#enhanced-state-legend-gradient)');
        
        // Legend scale
        const legendScale = d3.scaleLinear()
            .domain(colorScale.domain())
            .range([legendHeight, 0]);
        
        const legendAxis = d3.axisRight(legendScale)
            .ticks(5)
            .tickFormat(d => Math.round(d * 100) + '%');
        
        legend.append('g')
            .attr('transform', `translate(${legendWidth}, 0)`)
            .call(legendAxis);
        
        // Legend title
        legend.append('text')
            .attr('x', legendWidth / 2)
            .attr('y', -10)
            .style('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('font-weight', 'bold')
            .text('Similarity');
    }
    
    function handleEnhancedStateCellClick(data, i, j, stateA, stateB) {
        console.log(`Enhanced state cell clicked: ${stateA} vs ${stateB}`);
        // Could add interaction with other visualizations here
    }
    
    function updateEnhancedStateStats(data) {
        const statsElement = document.getElementById('heatmap-stats');
        
        if (!data || data.error) {
            statsElement.innerHTML = '<span class="text-muted">No data available</span>';
            return;
        }
        
        const numStates = data.total_states || 0;
        const totalComparisons = numStates * numStates;
        
        statsElement.innerHTML = `🗺️ ${numStates} states • ${totalComparisons} comparisons • Embedding-based similarity analysis`;
    }
    
    function getColorScale(colorScheme, matrix, heatmapType) {
        // Calculate data range
        const flatValues = matrix.flat();
        const minVal = Math.min(...flatValues);
        const maxVal = Math.max(...flatValues);
        
        let colorInterpolator;
        switch(colorScheme) {
            case 'reds': colorInterpolator = d3.interpolateReds; break;
            case 'viridis': colorInterpolator = d3.interpolateViridis; break;
            case 'plasma': colorInterpolator = d3.interpolatePlasma; break;
            case 'rdbu': colorInterpolator = d3.interpolateRdBu; break;
            default: colorInterpolator = d3.interpolateBlues;
        }
        
        // For diverging color schemes, center around midpoint
        if (colorScheme === 'rdbu') {
            const midPoint = (minVal + maxVal) / 2;
            return d3.scaleDiverging([minVal, midPoint, maxVal], colorInterpolator);
        } else {
            return d3.scaleSequential([minVal, maxVal], colorInterpolator);
        }
    }
    
    function createEnhancedTooltip() {
        // Remove existing tooltip if any
        d3.select('.heatmap-tooltip').remove();
        
        return d3.select('body').append('div')
            .attr('class', 'heatmap-tooltip')
            .style('position', 'absolute')
            .style('background', 'rgba(0,0,0,0.9)')
            .style('color', 'white')
            .style('padding', '12px')
            .style('border-radius', '6px')
            .style('font-size', '12px')
            .style('line-height', '1.4')
            .style('pointer-events', 'none')
            .style('opacity', 0)
            .style('box-shadow', '0 4px 8px rgba(0,0,0,0.2)');
    }
    
    function generateTooltipContent(data, i, j, value, heatmapType, rowLabels, colLabels) {
        let content = '';
        
        const rowLabel = rowLabels[i];
        const colLabel = colLabels[j];
        
        switch(heatmapType) {
            case 'cluster':
                const metadata = data.cluster_metadata;
                if (metadata && metadata[i] && metadata[j]) {
                    content = `
                        <strong>${rowLabel} ↔ ${colLabel}</strong><br/>
                        <span style="color: #ffd700;">Similarity: ${(value * 100).toFixed(1)}%</span><br/>
                        <div style="margin-top: 8px; font-size: 11px;">
                            <strong>Cluster A:</strong> ${metadata[i].size} standards<br/>
                            <strong>Cluster B:</strong> ${metadata[j].size} standards
                        </div>
                    `;
                }
                break;
            case 'topic':
                content = `
                    <strong>${rowLabel}</strong> in <strong>${colLabel}</strong><br/>
                    <span style="color: #28a745;">Coverage: ${value}%</span><br/>
                    <small>Educational topic adoption rate</small>
                `;
                break;
            case 'density':
                content = `
                    <strong>${rowLabel}</strong> - <strong>${colLabel}</strong><br/>
                    <span style="color: #17a2b8;">Standards Count: ${value}</span><br/>
                    <small>Number of educational standards</small>
                `;
                break;
            default: // state similarities
                content = `
                    <strong>${rowLabel} ↔ ${colLabel}</strong><br/>
                    <span style="color: #007bff;">Similarity: ${(value * 100).toFixed(1)}%</span><br/>
                    <small>Educational standards alignment</small>
                `;
        }
        
        return content;
    }
    
    function highlightRowCol(g, row, col, numRows, numCols, cellSize) {
        // Add row highlight
        g.append('rect')
            .attr('class', 'row-highlight')
            .attr('x', 0)
            .attr('y', row * cellSize)
            .attr('width', numCols * cellSize)
            .attr('height', cellSize)
            .attr('fill', 'none')
            .attr('stroke', '#ff6b35')
            .attr('stroke-width', 2)
            .attr('opacity', 0.7);
        
        // Add column highlight
        g.append('rect')
            .attr('class', 'col-highlight')
            .attr('x', col * cellSize)
            .attr('y', 0)
            .attr('width', cellSize)
            .attr('height', numRows * cellSize)
            .attr('fill', 'none')
            .attr('stroke', '#ff6b35')
            .attr('stroke-width', 2)
            .attr('opacity', 0.7);
    }
    
    function removeHighlights(g) {
        g.selectAll('.row-highlight, .col-highlight').remove();
    }
    
    function addSquareMatrixLabels(g, labels, cellSize, numItems) {
        // X-axis labels (top)
        g.selectAll('.label-x')
            .data(labels)
            .enter().append('text')
            .attr('class', 'label-x')
            .attr('x', (d, i) => (i + 0.5) * cellSize)
            .attr('y', -15)
            .attr('text-anchor', 'middle')
            .style('font-size', Math.min(10, cellSize * 0.3) + 'px')
            .style('font-weight', '500')
            .text(d => d.length > 8 ? d.substring(0, 8) + '...' : d);
        
        // Y-axis labels (left)
        g.selectAll('.label-y')
            .data(labels)
            .enter().append('text')
            .attr('class', 'label-y')
            .attr('x', -15)
            .attr('y', (d, i) => (i + 0.5) * cellSize + 4)
            .attr('text-anchor', 'end')
            .style('font-size', Math.min(10, cellSize * 0.3) + 'px')
            .style('font-weight', '500')
            .text(d => d.length > 8 ? d.substring(0, 8) + '...' : d);
    }
    
    function addAsymmetricMatrixLabels(g, rowLabels, colLabels, cellSize, numRows, numCols) {
        // Column labels (top) - can be rotated for better fit
        g.selectAll('.label-col')
            .data(colLabels)
            .enter().append('text')
            .attr('class', 'label-col')
            .attr('x', (d, i) => (i + 0.5) * cellSize)
            .attr('y', -15)
            .attr('text-anchor', colLabels[0].length > 4 ? 'start' : 'middle')
            .attr('transform', colLabels[0].length > 4 ? 
                (d, i) => `rotate(-45, ${(i + 0.5) * cellSize}, -15)` : '')
            .style('font-size', Math.min(9, cellSize * 0.25) + 'px')
            .style('font-weight', '500')
            .text(d => d.length > 12 ? d.substring(0, 12) + '...' : d);
        
        // Row labels (left)
        g.selectAll('.label-row')
            .data(rowLabels)
            .enter().append('text')
            .attr('class', 'label-row')
            .attr('x', -15)
            .attr('y', (d, i) => (i + 0.5) * cellSize + 4)
            .attr('text-anchor', 'end')
            .style('font-size', Math.min(9, cellSize * 0.25) + 'px')
            .style('font-weight', '500')
            .text(d => d.length > 15 ? d.substring(0, 15) + '...' : d);
    }
    
    function addColorLegend(svg, colorScale, heatmapType, x, y) {
        const legendHeight = 150;
        const legendWidth = 20;
        
        const legend = svg.append('g')
            .attr('transform', `translate(${x}, ${y})`);
        
        // Create gradient
        const defs = svg.append('defs');
        const gradient = defs.append('linearGradient')
            .attr('id', 'heatmap-legend-gradient')
            .attr('x1', '0%').attr('y1', '0%')
            .attr('x2', '0%').attr('y2', '100%');
        
        const domain = colorScale.domain();
        const steps = 10;
        
        for (let i = 0; i <= steps; i++) {
            const value = domain[0] + (domain[domain.length - 1] - domain[0]) * (i / steps);
            gradient.append('stop')
                .attr('offset', `${i * 10}%`)
                .attr('stop-color', colorScale(value));
        }
        
        // Draw legend rectangle
        legend.append('rect')
            .attr('width', legendWidth)
            .attr('height', legendHeight)
            .style('fill', 'url(#heatmap-legend-gradient)');
        
        // Add legend labels
        const legendScale = d3.scaleLinear()
            .domain(domain)
            .range([0, legendHeight]);
        
        const legendAxis = d3.axisRight(legendScale)
            .ticks(5)
            .tickFormat(d => {
                if (heatmapType === 'density') return d.toString();
                if (heatmapType === 'topic') return d + '%';
                return (d * 100).toFixed(0) + '%';
            });
        
        legend.append('g')
            .attr('transform', `translate(${legendWidth}, 0)`)
            .call(legendAxis);
        
        // Legend title
        const legendTitle = heatmapType === 'density' ? 'Count' :
                          heatmapType === 'topic' ? 'Coverage' : 'Similarity';
        
        legend.append('text')
            .attr('x', legendWidth / 2)
            .attr('y', -10)
            .attr('text-anchor', 'middle')
            .style('font-size', '11px')
            .style('font-weight', 'bold')
            .text(legendTitle);
    }
    
    function generateInfoText(data, heatmapType, numRows, numCols) {
        switch(heatmapType) {
            case 'cluster':
                return `${numRows} topic clusters analyzed • Click cells to explore relationships`;
            case 'topic':
                return `${numRows} topics across ${numCols} states • Coverage analysis`;
            case 'density':
                return `${numRows} grade levels × ${numCols} subjects • Standards distribution`;
            default:
                return `${numRows} states compared • Semantic similarity analysis`;
        }
    }
    
    // UI update functions for dual heat map types
    function updateHeatMapUI(heatmapType) {
        const titles = {
            topic: '📊 Topic Coverage Heat Map',
            enhanced_state: '🗺️ Enhanced State Analysis'
        };
        
        const subtitles = {
            topic: 'Topic clusters × states coverage analysis',
            enhanced_state: 'State-to-state similarity based on embedding analysis'
        };
        
        const descriptions = {
            topic: '<strong>Topic Coverage Analysis:</strong> Each row represents a clustered topic, each column represents a state. Color intensity shows how well each topic is covered in each state\'s standards.',
            enhanced_state: '<strong>Enhanced State Analysis:</strong> Advanced state comparison using direct embedding similarities instead of correlations. Shows how similar educational standards are between different states based on semantic analysis.'
        };
        
        // Update title and subtitle
        document.getElementById('heatmap-title').textContent = titles[heatmapType] || titles.topic;
        document.getElementById('heatmap-subtitle').textContent = subtitles[heatmapType] || subtitles.topic;
        
        // Update description
        document.getElementById('heatmap-description').innerHTML = 
            `<small class="text-muted">${descriptions[heatmapType] || descriptions.topic}</small>`;
    }
    
    function updateHeatMapStats(data, heatmapType) {
        if (heatmapType === 'topic') {
            updateTopicCoverageStats(data);
        } else if (heatmapType === 'enhanced_state') {
            updateEnhancedStateStats(data);
        }
    }
    
    
    function showHeatMapError(errorMessage) {
        const container = document.getElementById('heatmap');
        container.innerHTML = `
            <div class="text-center py-5">
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    <strong>Unable to load heat map</strong><br>
                    <small>${errorMessage}</small>
                </div>
                <button class="btn btn-outline-primary btn-sm" onclick="loadSimilarityMatrix()">
                    <i class="fas fa-redo"></i> Try Again
                </button>
            </div>
        `;
    }
    
    
    // Cross-visualization interaction functions
    function highlightClustersOnScatterPlot(clusterIds) {
        console.log('Highlighting clusters on scatter plot:', clusterIds);
        
        // Check if scatter plot exists
        const scatterPlot = document.getElementById('scatter-plot');
        if (!scatterPlot || !currentData.scatter) {
            console.warn('No scatter plot data available for highlighting');
            return;
        }
        
        // This would interact with the Plotly scatter plot to highlight specific clusters
        // For now, we'll show a visual indicator that clusters are selected
        const clusterInfo = document.getElementById('cluster-info');
        if (clusterInfo) {
            clusterInfo.innerHTML = `
                <div class="alert alert-info">
                    <strong>🎯 Clusters Highlighted:</strong> 
                    ${clusterIds.map(id => `Cluster ${id}`).join(', ')}
                    <button class="btn btn-sm btn-outline-secondary ms-2" onclick="clearClusterHighlight()">
                        Clear Highlight
                    </button>
                </div>
            `;
        }
        
        // Store the highlighted clusters for visual emphasis
        currentData.highlightedClusters = clusterIds;
        
        // If scatter plot is using Plotly, we could update the colors here
        try {
            const plotlyDiv = document.getElementById('scatter-plot');
            if (plotlyDiv && plotlyDiv.data) {
                // Update scatter plot colors to highlight selected clusters
                updateScatterPlotHighlight(clusterIds);
            }
        } catch (e) {
            console.log('Could not update scatter plot highlighting:', e);
        }
    }
    
    function filterScatterPlotByStates(stateList) {
        console.log('Filtering scatter plot by states:', stateList);
        
        // Store state filter
        currentData.stateFilter = stateList;
        
        // Show filter indicator
        const scatterInfo = document.getElementById('scatter-info');
        if (scatterInfo) {
            scatterInfo.innerHTML = `
                <span class="badge bg-primary me-2">
                    States filtered: ${stateList.join(', ')}
                    <button class="btn-close btn-close-white ms-1" onclick="clearStateFilter()"></button>
                </span>
                <small>Showing only standards from selected states</small>
            `;
        }
        
        // Reload visualization data with state filter
        // This would need to be implemented in the scatter plot loading logic
        // For now, we'll just reload the scatter plot
        if (currentData.scatter) {
            loadVisualizationData();
        }
    }
    
    function updateScatterPlotHighlight(clusterIds) {
        // This function would update the Plotly scatter plot to highlight specific clusters
        // Implementation depends on how the scatter plot data is structured
        console.log('Updating scatter plot highlight for clusters:', clusterIds);
        
        // Example implementation (would need to be adapted to actual scatter plot structure):
        /*
        const plotlyDiv = document.getElementById('scatter-plot');
        if (plotlyDiv && plotlyDiv.data && plotlyDiv.data[0]) {
            const updates = {
                'marker.opacity': plotlyDiv.data[0].marker.cluster.map(clusterId => 
                    clusterIds.includes(clusterId) ? 1.0 : 0.3
                )
            };
            
            Plotly.restyle(plotlyDiv, updates, [0]);
        }
        */
    }
    
    function showTopicStateDetails(topic, state, coverage) {
        // Show detailed information about topic coverage in a specific state
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">📚 Topic Coverage Details</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <h6><strong>Topic:</strong> ${topic}</h6>
                        <h6><strong>State:</strong> ${state}</h6>
                        <div class="progress mb-3">
                            <div class="progress-bar bg-success" style="width: ${coverage}%">
                                ${coverage}% Coverage
                            </div>
                        </div>
                        <p class="text-muted">
                            This topic has ${coverage}% adoption rate in ${state}, 
                            indicating ${coverage > 80 ? 'widespread' : coverage > 50 ? 'moderate' : 'limited'} 
                            coverage in the state's educational standards.
                        </p>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-primary" onclick="filterByTopicState('${topic}', '${state}')">
                            Filter Standards
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        const modalInstance = new bootstrap.Modal(modal);
        modalInstance.show();
        
        // Clean up modal after it's hidden
        modal.addEventListener('hidden.bs.modal', () => {
            modal.remove();
        });
    }
    
    // Utility functions for cross-visualization features
    window.clearClusterHighlight = function() {
        currentData.highlightedClusters = null;
        currentData.selectedClusters = null;
        
        const clusterInfo = document.getElementById('cluster-info');
        if (clusterInfo) {
            clusterInfo.innerHTML = '';
        }
        
        // Reset scatter plot highlighting
        updateScatterPlotHighlight([]);
    }
    
    window.clearStateFilter = function() {
        currentData.stateFilter = null;
        currentData.selectedStates = null;
        
        const scatterInfo = document.getElementById('scatter-info');
        if (scatterInfo) {
            scatterInfo.innerHTML = 'Loading...';
        }
        
        // Reload scatter plot without filter
        loadVisualizationData();
    }
    
    window.filterByTopicState = function(topic, state) {
        // Implementation for filtering by topic and state
        console.log(`Filtering by topic: ${topic}, state: ${state}`);
        
        // This would apply filters to the main visualization
        currentData.topicStateFilter = { topic, state };
        
        // Show filter in UI
        showToast(`Applied filter: ${topic} in ${state}`, 'info');
        
        // Close modal
        const modal = bootstrap.Modal.getInstance(document.querySelector('.modal'));
        if (modal) modal.hide();
    }
    
    // Export functions
    window.exportHeatMap = function(format = 'png') {
        const svg = document.querySelector('#heatmap svg');
        if (!svg) {
            alert('No heat map to export. Please generate a heat map first.');
            return;
        }
        
        if (format === 'svg') {
            const svgData = new XMLSerializer().serializeToString(svg);
            const blob = new Blob([svgData], {type: 'image/svg+xml'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'heatmap.svg';
            a.click();
        } else if (format === 'png') {
            // Convert SVG to PNG using canvas
            const svgData = new XMLSerializer().serializeToString(svg);
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            img.onload = function() {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.fillStyle = 'white';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
                
                canvas.toBlob(function(blob) {
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'heatmap.png';
                    a.click();
                });
            };
            
            img.src = 'data:image/svg+xml;base64,' + btoa(svgData);
        }
    }
    
    // Network graph visualization
    // Store network data globally for filtering
    let currentNetworkData = null;

    function cloneGraph(data) {
        return JSON.parse(JSON.stringify(data));
    }

    function renderNetworkGraph(data) {
        const graphData = cloneGraph(data);
        currentNetworkData = graphData;  // Store for filtering
        const container = document.getElementById('network-graph');
        container.innerHTML = '';
        
        if (graphData.error) {
            showNetworkError(graphData.error);
            return;
        }
        
        if (!graphData.nodes || graphData.nodes.length === 0) {
            showNetworkError('No network data available');
            return;
        }
        
        // Debug: Log concept nodes to check colors
        console.log('Concept nodes:', graphData.nodes.filter(n => n.type === 'concept').map(n => ({
            label: n.label,
            concept_type: n.concept_type,
            color: n.color,
            coverage: n.coverage_percentage
        })));
        
        // Get current filter states - using updated checkbox IDs
        const showCommon = document.getElementById('show-common-concepts')?.checked ?? true;
        const showSemiCommon = document.getElementById('show-semi-common-concepts')?.checked ?? true;
        const showStateSpecific = document.getElementById('show-state-specific')?.checked ?? true;
        const showStandards = document.getElementById('show-standards')?.checked ?? true;
        
        // Filter nodes based on checkboxes
        const filteredNodes = graphData.nodes.filter(node => {
            if (node.type === 'concept') {
                // Filter concept nodes by their concept_type
                if (node.concept_type === 'common' && !showCommon) return false;
                if (node.concept_type === 'semi_common' && !showSemiCommon) return false;
                if (node.concept_type === 'state_specific' && !showStateSpecific) return false;
            } else if (node.type === 'standard') {
                // Filter standard nodes
                if (!showStandards) return false;
            }
            return true;
        });
        
        const filteredEdges = graphData.edges.filter(edge => {
            // Only show edges if both source and target nodes are visible
            const sourceVisible = filteredNodes.find(n => n.id === edge.source.id || n.id === edge.source);
            const targetVisible = filteredNodes.find(n => n.id === edge.target.id || n.id === edge.target);
            
            return sourceVisible && targetVisible;
        });
        
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        // Create SVG
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        // Pre-position nodes to reduce initial chaos
        filteredNodes.forEach((node, i) => {
            const angle = (i / filteredNodes.length) * 2 * Math.PI;
            const radius = node.type === 'concept' ? 200 : 350;
            node.x = width / 2 + radius * Math.cos(angle);
            node.y = height / 2 + radius * Math.sin(angle);
        });
        
        // Create force simulation with balanced forces for stability
        const simulation = d3.forceSimulation(filteredNodes)
            .force('link', d3.forceLink(filteredEdges)
                .id(d => d.id)
                .distance(d => {
                    // Increased distances for better spread
                    if (d.type === 'implements') return 80;  // Was 40
                    if (d.type === 'secondary_relation') return 120;  // For secondary connections
                    if (d.type === 'strong_similarity') return 100;  // Was 50
                    if (d.type === 'moderate_similarity') return 140;  // Was 70
                    return 180;  // Was 90
                })
                .strength(d => {
                    // Reduced link strength for more flexibility
                    if (d.type === 'implements') return 0.8;  // Was 1.5
                    if (d.type === 'secondary_relation') return 0.3;  // Weak for secondary
                    return d.weight * 0.5 || 0.3;  // Was 0.5
                })
            )
            .force('charge', d3.forceManyBody()
                .strength(d => {
                    // Stronger repulsion for concept nodes to spread them out
                    if (d.type === 'concept') {
                        // Much stronger repulsion based on importance
                        if (d.concept_type === 'common') return -1200;  // Was -500
                        if (d.concept_type === 'semi_common') return -1000;  // Was -400
                        return -800;  // Was -300
                    }
                    return -150;  // Was -100, slightly stronger for standards too
                })
                .distanceMax(400)  // Increased from 200 for wider influence
            )
            .force('center', d3.forceCenter(width / 2, height / 2).strength(0.02))  // Reduced from 0.1
            .force('x', d3.forceX(width / 2).strength(0.01))  // Much gentler, was 0.05
            .force('y', d3.forceY(height / 2).strength(0.01))  // Much gentler, was 0.05
            .force('collision', d3.forceCollide()
                .radius(d => {
                    // Larger collision radius for concept nodes
                    if (d.type === 'concept') return d.size + 20;  // More space
                    return d.size + 5;
                })
                .strength(0.9)  // Increased from 0.7 for better collision prevention
                .iterations(2)  // Multiple iterations for better collision resolution
            )
            .force('boundary', () => {
                // Custom force to keep nodes within bounds
                filteredNodes.forEach(node => {
                    const margin = 50;
                    if (node.x < margin) node.x = margin;
                    if (node.x > width - margin) node.x = width - margin;
                    if (node.y < margin) node.y = margin;
                    if (node.y > height - margin) node.y = height - margin;
                });
            });
        
        // Configure simulation to settle faster
        simulation
            .alphaMin(0.01)  // Stop simulation sooner
            .alphaDecay(0.02)  // Faster cooling
            .velocityDecay(0.5);  // More damping to reduce movement
        
        // Create container groups
        const g = svg.append('g');
        
        // Add zoom behavior with initial zoom out to show all nodes
        const zoom = d3.zoom()
            .scaleExtent([0.3, 3])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });
        
        // Apply initial zoom to fit all nodes
        const initialScale = 0.5; // Start more zoomed out to see spread nodes
        svg.call(zoom);
        svg.call(zoom.transform, d3.zoomIdentity.scale(initialScale));
        
        // Store simulation reference for reset button
        window.networkSimulation = simulation;
        
        // Create tooltip
        const tooltip = d3.select('body').append('div')
            .attr('class', 'network-tooltip')
            .style('position', 'absolute')
            .style('visibility', 'hidden')
            .style('background', 'rgba(0, 0, 0, 0.8)')
            .style('color', 'white')
            .style('padding', '8px')
            .style('border-radius', '4px')
            .style('font-size', '12px')
            .style('max-width', '300px')
            .style('z-index', '1000');
        
        // Draw edges
        const links = g.selectAll('.link')
            .data(filteredEdges)
            .enter().append('line')
            .attr('class', 'link')
            .style('stroke', d => d.type === 'similar' ? '#4ECDC4' : '#95A5A6')
            .style('stroke-width', d => Math.max(1, d.weight * 3))
            .style('stroke-dasharray', d => d.style === 'dashed' ? '5,5' : '0')
            .style('opacity', 0.7);
        
        // Draw nodes
        const nodes = g.selectAll('.node')
            .data(filteredNodes)
            .enter().append('circle')
            .attr('class', 'node')
            .attr('r', d => d.size)
            .style('fill', d => d.color)
            .style('stroke', '#fff')
            .style('stroke-width', d => d.type === 'concept' ? 3 : 1)
            .style('cursor', 'pointer')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended))
            .on('mouseover', function(event, d) {
                // Highlight connected nodes and edges
                const connectedNodes = new Set();
                connectedNodes.add(d.id);
                
                links.style('opacity', link => {
                    if (link.source.id === d.id || link.target.id === d.id) {
                        connectedNodes.add(link.source.id);
                        connectedNodes.add(link.target.id);
                        return 1;
                    }
                    return 0.1;
                });
                
                nodes.style('opacity', node => connectedNodes.has(node.id) ? 1 : 0.3);
                
                // Show enhanced tooltip
                let tooltipContent = `<strong>${d.label}</strong><br/>`;
                if (d.type === 'concept') {
                    const typeLabel = d.concept_type === 'common' ? 'Common Concept' : 
                                     d.concept_type === 'semi_common' ? 'Semi-Common Concept' : 
                                     'State-Specific Concept';
                    tooltipContent += `
                        Type: ${typeLabel}<br/>
                        Coverage: ${d.coverage_percentage}% of states<br/>
                        States: ${d.implementing_states ? d.implementing_states.join(', ') : 'N/A'}<br/>
                        Standards: ${d.shown_standards || d.cluster_size} shown${d.total_standards && d.total_standards > d.shown_standards ? ` of ${d.total_standards} total` : ''}<br/>
                    `;
                    
                    // Add keywords if available
                    if (d.keywords && d.keywords.length > 0) {
                        tooltipContent += `<strong>Keywords:</strong> ${d.keywords.join(', ')}`;
                    }
                } else if (d.type === 'standard') {
                    const typeLabel = d.concept_type === 'common' ? 'Common Standard' : 
                                     d.concept_type === 'semi_common' ? 'Semi-Common Standard' : 
                                     d.concept_type === 'state_specific' ? 'State-Specific Standard' : 
                                     'Standard';
                    tooltipContent += `
                        Type: ${typeLabel}<br/>
                        State: ${d.state}<br/>
                        ${d.full_text ? d.full_text.substring(0, 150) + '...' : ''}
                    `;
                }
                
                tooltip.style('visibility', 'visible').html(tooltipContent);
            })
            .on('mousemove', function(event) {
                tooltip.style('top', (event.pageY - 10) + 'px')
                       .style('left', (event.pageX + 10) + 'px');
            })
            .on('mouseout', function() {
                // Reset highlighting
                links.style('opacity', 0.7);
                nodes.style('opacity', 1);
                tooltip.style('visibility', 'hidden');
            });
        
        // Add labels for concept nodes
        const labels = g.selectAll('.label')
            .data(filteredNodes.filter(d => d.type === 'concept'))
            .enter().append('text')
            .attr('class', 'label')
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em')
            .style('font-size', '11px')
            .style('font-weight', 'bold')
            .style('fill', '#333')
            .style('pointer-events', 'none')
            .text(d => d.label.length > 20 ? d.label.substring(0, 17) + '...' : d.label);
        
        // Update positions on simulation tick
        simulation.on('tick', () => {
            links
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            nodes
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
            
            labels
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
        
        // Store simulation for controls
        window.networkSimulation = simulation;
        
        // Drag functions
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.1).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = Math.max(20, Math.min(width - 20, event.x));
            d.fy = Math.max(20, Math.min(height - 20, event.y));
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            // Keep node fixed after dragging for better control
            // Uncomment the lines below to release the node after dragging
            // d.fx = null;
            // d.fy = null;
        }
    }
    
    function showNetworkError(errorMessage) {
        const container = document.getElementById('network-graph');
        container.innerHTML = `
            <div class="text-center py-5">
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    <strong>Unable to load network graph</strong><br>
                    <small>${errorMessage}</small>
                </div>
                <button class="btn btn-outline-primary btn-sm" onclick="loadNetworkGraph()">
                    <i class="fas fa-redo"></i> Try Again
                </button>
            </div>
        `;
    }
    
    function updateNetworkStats(data) {
        const statsElement = document.getElementById('network-stats');
        
        if (!data || data.error) {
            statsElement.innerHTML = '<span class="text-muted">No data available</span>';
            return;
        }
        
        // Use enhanced statistics if available
        if (data.core_concept_count !== undefined) {
            const commonCount = data.core_concept_count || 0;  // 'core' maps to 'common'
            const semiCommonCount = data.regional_concept_count || 0;  // 'regional' maps to 'semi_common'
            const stateSpecificCount = data.state_specific_count || 0;
            const standardCount = data.standard_count || 0;
            const edgeCount = data.total_edges || 0;
            const statesCount = data.states_represented || 0;
            
            statsElement.innerHTML = `📊 ${statesCount} states • Common: ${commonCount} • Semi-Common: ${semiCommonCount} • State-Specific: ${stateSpecificCount} • Standards: ${standardCount}`;
        } else {
            // Fallback to basic stats
            const conceptCount = data.concept_count || 0;
            const standardCount = data.standard_count || 0;
            const edgeCount = data.total_edges || 0;
            
            statsElement.innerHTML = `📊 ${conceptCount} concepts • ${standardCount} standards • ${edgeCount} connections`;
        }
    }
    
    // Network controls
    window.resetNetworkLayout = function() {
        // Reset both the simulation and the zoom
        if (window.networkSimulation) {
            // Reset node positions toward center
            window.networkSimulation.nodes().forEach(node => {
                node.x = undefined;
                node.y = undefined;
            });
            
            // Restart simulation with higher alpha for stronger repositioning
            window.networkSimulation.alpha(1).restart();
        }
        
        // Reset zoom to initial state
        const svg = d3.select('#network-graph svg');
        const g = svg.select('g');
        if (svg.node() && g.node()) {
            const zoom = d3.zoom();
            svg.call(zoom.transform, d3.zoomIdentity.scale(0.7));
        }
    }
    
    window.exportNetworkGraph = function(format = 'png') {
        const svg = document.querySelector('#network-graph svg');
        if (!svg) {
            alert('No network graph to export. Please generate a network graph first.');
            return;
        }
        
        // Simple export functionality (can be enhanced)
        const serializer = new XMLSerializer();
        const svgString = serializer.serializeToString(svg);
        const blob = new Blob([svgString], {type: 'image/svg+xml'});
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `network-graph.${format === 'png' ? 'svg' : format}`;
        a.click();
        URL.revokeObjectURL(url);
    }
    
    // API calls
    window.loadVisualizationData = async function loadVisualizationData() {
        const params = window.dashboardEmbeddings.buildParams(getFormData());
        const container = document.getElementById('scatter-plot');
        const endpoint = getEndpoint('visualizationData');
        if (!endpoint) {
            console.error('Visualization endpoint not configured');
            return;
        }

        showLoading('scatter-loading');
        console.log('Loading visualization data with params:', params.toString());

        try {
            const response = await fetch(`${endpoint}?${params.toString()}`);
            console.log('API response status:', response.status);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const response_data = await response.json();
            console.log('API response data:', response_data);
            
            // Validate response structure
            if (!response_data) {
                throw new Error('API returned null/undefined data');
            }
            
            if (response_data.error) {
                throw new Error(response_data.error);
            }
            
            // Handle nested data structure from APIResponseMixin
            const data = response_data.data || response_data;
            console.log('Extracted data:', data);
            
            // Check for expected data structure
            if (!data.hasOwnProperty('scatter_data')) {
                console.warn('Extracted data missing scatter_data property, available properties:', Object.keys(data));
                throw new Error('Invalid API response format: missing scatter_data');
            }
            
            currentData.scatter = data;
            
            // Add success feedback for users
            if (data.scatter_data && data.scatter_data.length > 0) {
                console.log(`Successfully loaded ${data.scatter_data.length} data points for visualization`);
            } else {
                console.log('API returned empty data set');
            }
            
            renderScatterPlot(data);
            
        } catch (error) {
            console.error('Error loading visualization data:', error);
            
            // Provide detailed error information to users
            let errorMessage = 'Error loading visualization data';
            if (error.message) {
                errorMessage += ': ' + error.message;
            }
            
            container.innerHTML = `
                <div class="alert alert-danger">
                    <h6><i class="fas fa-exclamation-triangle"></i> Loading Error</h6>
                    <p>${errorMessage}</p>
                    <details class="mt-2">
                        <summary class="small text-muted" style="cursor: pointer;">Technical Details</summary>
                        <small class="text-muted mt-1 d-block">
                            Request URL: ${endpoint}?${params.toString()}<br>
                            Error: ${error.name || 'Unknown'}<br>
                            Message: ${error.message || 'No details available'}
                        </small>
                    </details>
                    <button class="btn btn-sm btn-outline-danger mt-2" onclick="loadVisualizationData()">
                        <i class="fas fa-redo"></i> Retry
                    </button>
                </div>
            `;
        } finally {
            hideLoading('scatter-loading');
        }
    }
    
    // Heat map loading for multiple types
    window.loadSimilarityMatrix = async function loadSimilarityMatrix() {
        const heatmapType = document.getElementById('heatmap-type')?.value || 'topic';
        await loadHeatMapData(heatmapType);
    }
    
    window.loadHeatMapData = async function loadHeatMapData(heatmapType = 'topic') {
        const params = window.dashboardEmbeddings.buildParams(getFormData());
        showLoading('heatmap-loading');
        
        // Update UI based on heat map type
        updateHeatMapUI(heatmapType);
        
        try {
            let url;
            const additionalParams = {};
            
            // Select appropriate API endpoint based on type
            if (heatmapType === 'topic') {
                url = getEndpoint('clusterMatrix');
                additionalParams.cluster_size = params.get('cluster_size') || '5';
                additionalParams.epsilon = params.get('epsilon') || '0.5';
            } else if (heatmapType === 'enhanced_state') {
                url = getEndpoint('enhancedMatrix');
                additionalParams.matrix_type = 'enhanced_state';
            }

            if (!url) {
                throw new Error('Heat map endpoint not configured');
            }
            
            // Add specific parameters to the request
            Object.keys(additionalParams).forEach(key => {
                params.set(key, additionalParams[key]);
            });
            
            const response = await fetch(`${url}?${params.toString()}`);
            const response_data = await response.json();
            
            if (response_data.success && response_data.data) {
                const data = response_data.data;
                
                // Store current data for export and interactions
                currentData.heatmap = data;
                currentData.heatmapType = heatmapType;
                
                // Render appropriate heat map type
                if (heatmapType === 'topic') {
                    renderTopicCoverageHeatmap(data);
                    updateTopicCoverageStats(data);
                } else if (heatmapType === 'enhanced_state') {
                    renderEnhancedStateHeatmap(data);
                    updateEnhancedStateStats(data);
                }
            } else {
                console.error('Error loading heat map:', response_data.error);
                showHeatMapError(response_data.error || 'Failed to load heat map data');
                updateHeatMapStats(null, heatmapType);
            }
        } catch (error) {
            console.error('Error loading heat map:', error);
            showHeatMapError(`Failed to load heat map: ${error.message}`);
            updateHeatMapStats(null, heatmapType);
        } finally {
            hideLoading('heatmap-loading');
        }
    }
    
    async function loadNetworkGraph() {
        const params = window.dashboardEmbeddings.buildParams(getFormData());
        const endpoint = getEndpoint('networkGraph');
        if (!endpoint) {
            console.error('Network graph endpoint not configured');
            return;
        }
        showLoading('network-loading');
        
        try {
            const response = await fetch(`${endpoint}?${params.toString()}`);
            const response_data = await response.json();
            
            if (response_data.success && response_data.data) {
                const data = response_data.data;
                currentData.network = data;
                const canonical = cloneGraph(data);
                if (window.dashboardEmbeddings && typeof window.dashboardEmbeddings.setNetworkData === 'function') {
                    window.dashboardEmbeddings.setNetworkData(cloneGraph(canonical));
                }
                renderNetworkGraph(canonical);
                updateNetworkStats(data);
            } else {
                console.error('Error loading network graph:', response_data.error);
                showNetworkError(response_data.error || 'Failed to load network graph');
            }
        } catch (error) {
            console.error('Error loading network graph:', error);
            showNetworkError(`Failed to load network graph: ${error.message}`);
        } finally {
            hideLoading('network-loading');
        }
    }
    
    // Handle network graph filter changes
    function handleNetworkFilterChange() {
        let baseData = null;
        if (window.dashboardEmbeddings && typeof window.dashboardEmbeddings.getNetworkData === 'function') {
            baseData = window.dashboardEmbeddings.getNetworkData();
        }
        if (!baseData && currentNetworkData) {
            baseData = cloneGraph(currentNetworkData);
        }
        if (baseData) {
            renderNetworkGraph(baseData);
            updateFilteredNetworkStats();
        }
    }
    
    function updateFilteredNetworkStats() {
        if (!currentNetworkData) return;
        
        const showCommon = document.getElementById('show-common-concepts')?.checked ?? true;
        const showSemiCommon = document.getElementById('show-semi-common-concepts')?.checked ?? true;
        const showStateSpecific = document.getElementById('show-state-specific')?.checked ?? true;
        const showStandards = document.getElementById('show-standards')?.checked ?? true;
        
        // Count filtered nodes
        const filteredNodes = currentNetworkData.nodes.filter(node => {
            if (node.type === 'concept') {
                if (node.concept_type === 'common' && !showCommon) return false;
                if (node.concept_type === 'semi_common' && !showSemiCommon) return false;
                if (node.concept_type === 'state_specific' && !showStateSpecific) return false;
            }
            if (node.type === 'standard' && !showStandards) return false;
            return true;
        });
        
        // Count filtered edges
        const filteredEdges = currentNetworkData.edges.filter(edge => {
            const sourceVisible = filteredNodes.find(n => n.id === edge.source.id || n.id === edge.source);
            const targetVisible = filteredNodes.find(n => n.id === edge.target.id || n.id === edge.target);
            
            return sourceVisible && targetVisible;
        });
        
        const commonCount = filteredNodes.filter(n => n.type === 'concept' && n.concept_type === 'common').length;
        const semiCommonCount = filteredNodes.filter(n => n.type === 'concept' && n.concept_type === 'semi_common').length;
        const stateSpecificCount = filteredNodes.filter(n => n.type === 'concept' && n.concept_type === 'state_specific').length;
        const standardCount = filteredNodes.filter(n => n.type === 'standard').length;
        const edgeCount = filteredEdges.length;
        
        const statsElement = document.getElementById('network-stats');
        statsElement.innerHTML = `📊 Common: ${commonCount} • Semi-Common: ${semiCommonCount} • State-Specific: ${stateSpecificCount} • Standards: ${standardCount} • Edges: ${edgeCount}`;
    }
    
    // Event listeners
    document.getElementById('update-viz').addEventListener('click', function() {
        console.log('Update button clicked');
        
        // Load data for active tab
        const activeTab = document.querySelector('#viz-tabs .nav-link.active');
        console.log('Active tab element:', activeTab);
        
        if (!activeTab) {
            console.error('No active tab found in viz-tabs');
            return;
        }
        
        const activeTabId = activeTab.id;
        console.log('Active tab ID:', activeTabId);
        
        switch(activeTabId) {
            case 'scatter-tab':
                console.log('Loading scatter plot data');
                loadVisualizationData();
                break;
            case 'heatmap-tab':
                console.log('Loading heatmap data');
                loadSimilarityMatrix();
                break;
            case 'network-tab':
                console.log('Loading network graph data');
                loadNetworkGraph();
                break;
            case 'themes-tab':
                console.log('Loading themes data');
                loadThemeCoverage();
                break;
            default:
                console.warn('Unknown active tab:', activeTabId);
        }
    });
    
    // Initialize shared semantic search module
    const searchRoot = document.querySelector('#search-pane [data-semantic-search]');
    if (searchRoot && window.dashboardSemanticSearch) {
        const searchEndpoint = getEndpoint('semanticSearch');
        window.dashboardSemanticSearch.init({
            root: searchRoot,
            searchUrl: searchEndpoint || '',
            csrfToken: getCookie('csrftoken'),
            getFilters: () => {
                const formData = getFormData();
                return {
                    grade_level: formData.grade_level,
                    subject_area: formData.subject_area
                };
            },
            mode: 'view'
        });
    }

    // Heat map control event listeners
    document.getElementById('heatmap-type').addEventListener('change', function() {
        const heatmapType = this.value;
        console.log('Heat map type changed to:', heatmapType);
        
        // Clear current heat map data to force reload
        currentData.heatmap = null;
        
        // Load new heat map data
        loadHeatMapData(heatmapType);
    });
    
    document.getElementById('heatmap-colorscheme').addEventListener('change', function() {
        console.log('Color scheme changed to:', this.value);
        
        // If we have current data, re-render with new color scheme
        if (currentData.heatmap && currentData.heatmapType) {
            if (currentData.heatmapType === 'topic') {
                renderTopicCoverageHeatmap(currentData.heatmap);
            } else if (currentData.heatmapType === 'enhanced_state') {
                renderEnhancedStateHeatmap(currentData.heatmap);
            }
        }
    });
    
    // Tab change handlers
    document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', function(e) {
            const target = e.target.getAttribute('data-bs-target');
            
            // Load data when tab becomes active if not already loaded
            switch(target) {
                case '#scatter-pane':
                    if (!currentData.scatter) loadVisualizationData();
                    break;
                case '#heatmap-pane':
                    if (!currentData.heatmap) loadSimilarityMatrix();
                    break;
                case '#network-pane':
                    if (!currentData.network) loadNetworkGraph();
                    break;
                case '#themes-pane':
                    if (!currentData.themes) loadThemeCoverage();
                    break;
            }
        });
    });
    
    // Initialize (optionally suppressed for custom-cluster dashboards)
    if (!window.dashboardEmbeddings || window.dashboardEmbeddings.shouldAutoInit()) {
        loadVisualizationData();
    }
})();
