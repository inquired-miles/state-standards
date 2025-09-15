(function () {
    if (window.dashboardCustomClusters) {
        return;
    }

    const state = {
        endpoints: {},
        initialized: false,
        csrfToken: null,
        modal: null,
        searchInstance: null,
        lastSearchPayload: null,
    };

    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) {
            return parts.pop().split(';').shift();
        }
        return null;
    }

    function fetchJSON(url, options = {}) {
        const defaultHeaders = {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        };
        const merged = Object.assign({
            headers: defaultHeaders,
            credentials: 'same-origin'
        }, options);

        if (options.headers) {
            merged.headers = Object.assign({}, defaultHeaders, options.headers);
        }

        return fetch(url, merged).then(async response => {
            let data;
            try {
                data = await response.json();
            } catch (err) {
                data = {};
            }
            if (!response.ok) {
                const message = data.error || data.message || `Request failed: ${response.status}`;
                const error = new Error(message);
                error.details = data;
                throw error;
            }
            return data;
        });
    }

    function renderClusterList(payload) {
        const container = document.getElementById('cluster-list');
        if (!container) return;
        container.innerHTML = '';
        const clusters = (payload && payload.data && payload.data.clusters) || [];
        if (!clusters.length) {
            container.innerHTML = '<div class="text-muted small">No custom clusters yet.</div>';
            return;
        }
        clusters.forEach(cluster => {
            const item = document.createElement('button');
            item.type = 'button';
            item.className = 'list-group-item list-group-item-action';
            item.textContent = `${cluster.name} (${cluster.standards_count})`;
            item.addEventListener('click', () => loadClusterDetail(cluster.id));
            container.appendChild(item);
        });
    }

    function renderClusterDetail(payload) {
        const cluster = payload && payload.data;
        if (!cluster) {
            const title = document.getElementById('cluster-detail-title');
            const placeholder = document.getElementById('cluster-placeholder');
            const summaryEl = document.getElementById('cluster-summary');
            const vizContainer = document.getElementById('cluster-visualizations');
            if (title) title.textContent = 'Cluster not found';
            if (placeholder) {
                placeholder.classList.remove('d-none');
                placeholder.textContent = 'Unable to load cluster details.';
            }
            if (summaryEl) {
                summaryEl.classList.add('d-none');
                summaryEl.innerHTML = '';
            }
            if (vizContainer) {
                vizContainer.classList.add('d-none');
            }
            return;
        }
        handleVisualizationPanel(cluster);
    }

    function renderReportList(payload) {
        const container = document.getElementById('report-list');
        if (!container) return;
        container.innerHTML = '';
        const reports = (payload && payload.data && payload.data.reports) || [];
        if (!reports.length) {
            container.innerHTML = '<div class="text-muted small">No reports yet.</div>';
            return;
        }
        reports.forEach(report => {
            const item = document.createElement('button');
            item.type = 'button';
            item.className = 'list-group-item list-group-item-action';
            item.textContent = `${report.title}`;
            item.addEventListener('click', () => loadReportDetail(report.id));
            container.appendChild(item);
        });
    }

    function renderReportDetail(payload) {
        const wrapper = document.getElementById('report-detail');
        const title = document.getElementById('report-detail-title');
        const body = document.getElementById('report-detail-body');
        if (!wrapper || !title || !body) return;
        const report = payload && payload.data;
        if (!report) {
            wrapper.hidden = false;
            title.textContent = 'Report not found';
            body.innerHTML = '<div class="text-danger">Unable to load report details.</div>';
            return;
        }
        wrapper.hidden = false;
        title.textContent = report.title;
        const rows = (report.clusters || []).map(entry => {
            const summary = JSON.stringify(entry.summary || {}, null, 2);
            return `<div class="mb-3"><h6>${entry.cluster_name}</h6><pre class="bg-light p-2 rounded">${summary}</pre></div>`;
        }).join('');
        body.innerHTML = rows || '<div class="text-muted">No clusters have been added to this report yet.</div>';
    }

    function loadClusterList() {
        fetchJSON(state.endpoints.clustersEndpoint)
            .then(renderClusterList)
            .catch(err => console.error(err));
    }

    function loadClusterDetail(clusterId) {
        fetchJSON(`${state.endpoints.clusterDetailBase}${clusterId}/`)
            .then(renderClusterDetail)
            .catch(err => console.error(err));
    }

    function loadReportList() {
        fetchJSON(state.endpoints.reportsEndpoint)
            .then(renderReportList)
            .catch(err => console.error(err));
    }

    function loadReportDetail(reportId) {
        fetchJSON(`${state.endpoints.reportDetailBase}${reportId}/`)
            .then(renderReportDetail)
            .catch(err => console.error(err));
    }

    function handleCreateCluster(selected) {
        const modalEl = document.getElementById('cluster-builder-modal');
        if (!modalEl) return;

        const titleInput = modalEl.querySelector('#cluster-title-input');
        const descriptionInput = modalEl.querySelector('#cluster-description-input');
        const title = (titleInput?.value || '').trim();
        const description = (descriptionInput?.value || '').trim();

        if (!title) {
            alert('Please provide a title for the custom cluster.');
            return;
        }

        if (!selected.length) {
            alert('Select at least one standard before saving the cluster.');
            return;
        }

        const similarityMap = selected.reduce((acc, item) => {
            if (typeof item.similarity_score === 'number') {
                acc[item.id] = item.similarity_score;
            }
            return acc;
        }, {});

        const payload = {
            title,
            description,
            standard_ids: selected.map(item => item.id),
            similarity_map: similarityMap,
            search_context: state.lastSearchPayload || {}
        };

        const headers = {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest',
            'X-CSRFToken': state.csrfToken || getCookie('csrftoken')
        };

        fetchJSON(state.endpoints.clustersEndpoint, {
            method: 'POST',
            headers,
            body: JSON.stringify(payload)
        }).then(data => {
            if (state.modal) {
                state.modal.hide();
            }
            loadClusterList();
            const cluster = data.data || data;
            if (cluster && cluster.id) {
                loadClusterDetail(cluster.id);
            }
        }).catch(err => {
            console.error(err);
            alert(err.message || 'Failed to create cluster.');
        });
    }

    function openClusterBuilder() {
        const modalEl = document.getElementById('cluster-builder-modal');
        if (!modalEl || !window.bootstrap) {
            console.warn('Cluster builder modal or Bootstrap not available');
            return;
        }

        if (!state.modal) {
            state.modal = new window.bootstrap.Modal(modalEl);
        }

        if (!state.searchInstance && window.dashboardSemanticSearch) {
            const searchRoot = modalEl.querySelector('[data-semantic-search]');
            state.searchInstance = window.dashboardSemanticSearch.init({
                root: searchRoot,
                searchUrl: state.endpoints.semanticSearchEndpoint,
                csrfToken: state.csrfToken || getCookie('csrftoken'),
                mode: 'select',
                onSearchComplete: (_data, payload) => {
                    state.lastSearchPayload = payload;
                },
                onConfirmSelection: handleCreateCluster
            });
        }

        const titleInput = modalEl.querySelector('#cluster-title-input');
        const descriptionInput = modalEl.querySelector('#cluster-description-input');
        if (titleInput) titleInput.value = '';
        if (descriptionInput) descriptionInput.value = '';

        if (state.searchInstance && typeof state.searchInstance.clearSelection === 'function') {
            state.searchInstance.clearSelection();
        }

        state.lastSearchPayload = null;

        state.modal.show();
    }

    window.dashboardCustomClusters = {
        init(endpoints) {
            if (state.initialized) {
                return;
            }
            state.initialized = true;
            state.endpoints = Object.assign({}, endpoints);
            state.csrfToken = endpoints.csrfToken || getCookie('csrftoken');
            loadClusterList();
            loadReportList();

            const trigger = document.getElementById('new-cluster-btn');
            if (trigger) {
                trigger.addEventListener('click', openClusterBuilder);
            }
        },
        refresh() {
            loadClusterList();
            loadReportList();
        }
    };
})();
    function handleVisualizationPanel(cluster) {
        const title = document.getElementById('cluster-detail-title');
        const placeholder = document.getElementById('cluster-placeholder');
        const summaryEl = document.getElementById('cluster-summary');
        const vizContainer = document.getElementById('cluster-visualizations');
        if (!title || !summaryEl || !vizContainer) return;

        title.textContent = cluster.name || 'Custom Cluster';
        if (placeholder) {
            placeholder.classList.add('d-none');
        }

        const members = (cluster.members || []).map(member => {
            return `<li><strong>${member.code}</strong> – ${member.title || 'Untitled'} <span class="text-muted">(${member.state || 'N/A'})</span></li>`;
        }).join('');
        const summary = cluster.coverage_summary || {};

        summaryEl.classList.remove('d-none');
        summaryEl.innerHTML = `
            <p class="text-muted">${cluster.description || 'No description provided.'}</p>
            <h6>Members (${cluster.standards_count})</h6>
            <ul class="ps-3">${members || '<li>No standards yet.</li>'}</ul>
            <h6 class="mt-3">Coverage Snapshot</h6>
            <pre class="bg-light p-2 rounded">${JSON.stringify(summary, null, 2)}</pre>
        `;

        const standardIds = (cluster.members || []).map(member => member.id);
        if (!standardIds.length) {
            vizContainer.classList.add('d-none');
            if (placeholder) {
                placeholder.classList.remove('d-none');
                placeholder.textContent = 'Add standards to this cluster to see visual analytics.';
            }
            if (summaryEl) {
                summaryEl.classList.add('d-none');
                summaryEl.innerHTML = '';
            }
            if (window.dashboardEmbeddings) {
                window.dashboardEmbeddings.setStandardFilter([]);
            }
            return;
        }

        vizContainer.classList.remove('d-none');

        if (window.dashboardEmbeddings) {
            window.dashboardEmbeddings.setStandardFilter(standardIds);
            if (typeof window.loadVisualizationData === 'function') {
                window.loadVisualizationData();
            }
            if (typeof window.loadHeatMapData === 'function') {
                window.loadHeatMapData('topic');
            }
            if (typeof window.loadNetworkGraph === 'function') {
                window.loadNetworkGraph();
            }
        }
    }
