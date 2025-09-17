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
        clusterSummaries: new Map(),
        clusterDetails: new Map(),
        selectedClusters: new Map(),
        activeClusterId: null,
        activeFilters: {
            gradeLevel: '',
            subjectArea: ''
        },
        vizMode: '2d',
        modalMode: 'create',
        editingClusterId: null
    };

    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) {
            return parts.pop().split(';').shift();
        }
        return null;
    }

    function buildEndpoint(base, append = '') {
        if (!base) {
            return '';
        }

        const normalizedBase = String(base).trim();
        if (!append) {
            return normalizedBase;
        }

        const baseWithoutTrailingSlash = normalizedBase.replace(/\/+$/, '');
        const normalizedAppend = String(append).replace(/^\/+/, '');
        return `${baseWithoutTrailingSlash}/${normalizedAppend}`;
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

    function removeClusterFromState(clusterId) {
        if (!clusterId) {
            return;
        }

        state.clusterDetails.delete(clusterId);
        state.clusterSummaries.delete(clusterId);
        state.selectedClusters.delete(clusterId);

        const selectedArray = getSelectedClustersArray();

        if (state.activeClusterId === clusterId) {
            state.activeClusterId = selectedArray.length ? selectedArray[0].id : null;
        }
        if (selectedArray.length === 0) {
            renderClusterDetail(null);
        } else if (selectedArray.length === 1) {
            renderClusterDetail(selectedArray[0]);
        } else {
            renderMultiClusterSummary(selectedArray);
        }

        updateSelectionUI();
        refreshVisualizationForSelection();
    }

    function performClusterDeletion(clusterId) {
        if (!clusterId) {
            return Promise.resolve();
        }

        const url = buildEndpoint(state.endpoints.clusterDetailBase, `${clusterId}/`);
        if (!url) {
            console.error('Cluster detail endpoint not configured');
            return Promise.reject(new Error('Cluster endpoint not configured'));
        }

        const headers = {
            'X-Requested-With': 'XMLHttpRequest',
            'X-CSRFToken': state.csrfToken || getCookie('csrftoken')
        };

        return fetchJSON(url, {
            method: 'DELETE',
            headers
        }).then(payload => {
            removeClusterFromState(clusterId);
            loadClusterList();
            return payload;
        }).catch(err => {
            console.error('Failed to delete cluster', err);
            throw err;
        });
    }

    function renderClusterList(payload) {
        const container = document.getElementById('cluster-list');
        if (!container) return;
        container.innerHTML = '';

        const clusters = (payload && payload.data && payload.data.clusters) || [];
        state.clusterSummaries.clear();

        if (!clusters.length) {
            container.innerHTML = '<div class="text-muted small">No custom clusters yet.</div>';
            return;
        }

        clusters.forEach(cluster => {
            state.clusterSummaries.set(cluster.id, cluster);

            const listItem = document.createElement('div');
            listItem.className = 'list-group-item list-group-item-action d-flex align-items-center justify-content-between';

            const formCheck = document.createElement('div');
            formCheck.className = 'form-check flex-grow-1';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.className = 'form-check-input position-static me-2';
            checkbox.id = `cluster-select-${cluster.id}`;
            checkbox.dataset.clusterId = cluster.id;
            checkbox.checked = state.selectedClusters.has(cluster.id);

            const label = document.createElement('label');
            label.className = 'form-check-label d-flex justify-content-between align-items-center w-100';
            label.htmlFor = checkbox.id;
            label.innerHTML = `
                <span class="fw-semibold">${cluster.name}</span>
                <span class="text-muted small ms-2">${cluster.standards_count}</span>
            `;

            formCheck.appendChild(checkbox);
            formCheck.appendChild(label);

            const viewButton = document.createElement('button');
            viewButton.type = 'button';
            viewButton.className = 'btn btn-sm btn-outline-secondary ms-2';
            viewButton.textContent = 'View';
            viewButton.addEventListener('click', event => {
                event.preventDefault();
                if (!checkbox.checked) {
                    checkbox.checked = true;
                }
                handleClusterSelection(cluster.id, true).then(() => {
                    focusOnCluster(cluster.id);
                });
            });

            checkbox.addEventListener('change', event => {
                handleClusterSelection(cluster.id, event.target.checked);
            });

            listItem.addEventListener('click', event => {
                if (
                    event.target === checkbox ||
                    event.target === viewButton ||
                    (event.target.closest && event.target.closest('button') === viewButton) ||
                    (event.target.closest && event.target.closest('.form-check'))
                ) {
                    return;
                }
                event.preventDefault();
                checkbox.checked = !checkbox.checked;
                handleClusterSelection(cluster.id, checkbox.checked);
            });

            listItem.appendChild(formCheck);
            listItem.appendChild(viewButton);
            container.appendChild(listItem);
        });

        updateSelectionUI();
    }

    function renderClusterDetail(cluster) {
        if (!cluster) {
            const title = document.getElementById('cluster-detail-title');
            const placeholder = document.getElementById('cluster-placeholder');
            const summaryEl = document.getElementById('cluster-summary');
            const vizContainer = document.getElementById('cluster-visualizations');
            if (title) title.textContent = 'Select a cluster to begin';
            if (placeholder) {
                placeholder.classList.remove('d-none');
                placeholder.textContent = 'Choose one or more clusters from the list to explore their coverage and relationships.';
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
        const url = buildEndpoint(state.endpoints.clustersEndpoint);
        if (!url) {
            console.error('Clusters endpoint not configured');
            return;
        }
        fetchJSON(url)
            .then(renderClusterList)
            .catch(err => console.error(err));
    }

    function loadClusterDetail(clusterId) {
        return ensureClusterDetail(clusterId).then(cluster => {
            if (!cluster) {
                renderClusterDetail(null);
                return null;
            }
            state.selectedClusters.set(clusterId, cluster);
            state.activeClusterId = clusterId;
            updateSelectionUI();
            renderClusterDetail(cluster);
            refreshVisualizationForSelection();
            return cluster;
        }).catch(err => {
            console.error(err);
            renderClusterDetail(null);
            return null;
        });
    }

    function loadReportList() {
        const url = buildEndpoint(state.endpoints.reportsEndpoint);
        if (!url) {
            console.error('Report list endpoint not configured');
            return;
        }
        fetchJSON(url)
            .then(renderReportList)
            .catch(err => console.error(err));
    }

    function loadReportDetail(reportId) {
        const url = buildEndpoint(state.endpoints.reportDetailBase, `${reportId}/`);
        if (!url) {
            console.error('Report detail endpoint not configured');
            return;
        }
        fetchJSON(url)
            .then(renderReportDetail)
            .catch(err => console.error(err));
    }

    function ensureClusterDetail(clusterId) {
        if (!clusterId) {
            return Promise.resolve(null);
        }
        if (state.clusterDetails.has(clusterId)) {
            return Promise.resolve(state.clusterDetails.get(clusterId));
        }

        const url = buildEndpoint(state.endpoints.clusterDetailBase, `${clusterId}/`);
        if (!url) {
            console.error('Cluster detail endpoint not configured');
            return Promise.resolve(null);
        }

        return fetchJSON(url).then(payload => {
            const cluster = payload && payload.data ? payload.data : null;
            if (cluster) {
                state.clusterDetails.set(cluster.id, cluster);
            }
            return cluster;
        });
    }

    function getSelectedClustersArray() {
        return Array.from(state.selectedClusters.values());
    }

    function updateSelectionUI() {
        const counter = document.getElementById('selected-count');
        const compareBtn = document.getElementById('compare-selected-btn');
        const count = state.selectedClusters.size;
        if (counter) {
            counter.textContent = String(count);
        }
        if (compareBtn) {
            compareBtn.disabled = count === 0;
        }
    }

    function updateVizModeButtons() {
        const buttons = document.querySelectorAll('[data-viz-mode]');
        buttons.forEach(button => {
            if (button.dataset.vizMode === state.vizMode) {
                button.classList.add('active');
                button.classList.remove('btn-outline-primary');
                button.classList.add('btn-primary');
            } else {
                button.classList.remove('active');
                button.classList.add('btn-outline-primary');
                button.classList.remove('btn-primary');
            }
        });
    }

    function setVizMode(mode, { refresh = true } = {}) {
        if (!mode || (mode !== '2d' && mode !== '3d')) {
            return;
        }
        if (state.vizMode === mode) {
            if (refresh) {
                refreshVisualizationForSelection();
            }
            return;
        }
        state.vizMode = mode;
        updateVizModeButtons();
        syncHiddenFilters();
        if (refresh) {
            refreshVisualizationForSelection();
        }
    }

    function syncHiddenFilters() {
        const mainGrade = document.getElementById('grade-level');
        const mainSubject = document.getElementById('subject-area');
        const vizSelect = document.getElementById('viz-mode');
        if (mainGrade) {
            mainGrade.value = state.activeFilters.gradeLevel || '';
        }
        if (mainSubject) {
            mainSubject.value = state.activeFilters.subjectArea || '';
        }
        if (vizSelect) {
            vizSelect.value = state.vizMode || '2d';
        }
    }

    function handleFilterChange() {
        const gradeSelect = document.getElementById('cluster-grade-filter');
        const subjectSelect = document.getElementById('cluster-subject-filter');
        state.activeFilters.gradeLevel = gradeSelect ? gradeSelect.value : '';
        state.activeFilters.subjectArea = subjectSelect ? subjectSelect.value : '';
        syncHiddenFilters();
        refreshVisualizationForSelection();
    }

    function buildManualClusterMap(clusters) {
        const labelMap = {};
        clusters.forEach(cluster => {
            const memberIds = (cluster.members || []).map(member => member.id);
            if (memberIds.length) {
                labelMap[cluster.id] = memberIds;
            }
        });
        return labelMap;
    }

    function refreshVisualizationForSelection() {
        const clusters = getSelectedClustersArray();
        const placeholder = document.getElementById('cluster-placeholder');
        const vizContainer = document.getElementById('cluster-visualizations');
        const vizModeControls = document.getElementById('viz-mode-controls');

        const standardIds = clusters.flatMap(cluster => (cluster.members || []).map(member => member.id));

        if (!standardIds.length) {
            if (vizContainer) {
                vizContainer.classList.add('d-none');
            }
            if (vizModeControls) {
                vizModeControls.classList.add('d-none');
            }
            if (placeholder) {
                placeholder.classList.remove('d-none');
                placeholder.textContent = 'Select clusters that contain standards with embeddings to load visual analytics.';
            }
            if (window.dashboardEmbeddings) {
                window.dashboardEmbeddings.setStandardFilter([]);
                if (typeof window.dashboardEmbeddings.setManualClusterMap === 'function') {
                    window.dashboardEmbeddings.setManualClusterMap({});
                }
            }
            return;
        }

        if (placeholder) {
            placeholder.classList.add('d-none');
        }
        if (vizContainer) {
            vizContainer.classList.remove('d-none');
        }
        if (vizModeControls) {
            vizModeControls.classList.remove('d-none');
        }

        syncHiddenFilters();
        updateVizModeButtons();

        const manualMap = buildManualClusterMap(clusters);

        if (window.dashboardEmbeddings) {
            window.dashboardEmbeddings.setStandardFilter([]);
            if (typeof window.dashboardEmbeddings.setManualClusterMap === 'function') {
                window.dashboardEmbeddings.setManualClusterMap(manualMap);
            }
        }

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

    function renderMultiClusterSummary(selectedClusters) {
        const title = document.getElementById('cluster-detail-title');
        const summaryEl = document.getElementById('cluster-summary');
        const placeholder = document.getElementById('cluster-placeholder');

        if (!title || !summaryEl) {
            return;
        }

        const count = selectedClusters.length;
        title.textContent = count === 2 ? 'Comparing 2 Clusters' : `Comparing ${count} Clusters`;

        if (placeholder) {
            placeholder.classList.add('d-none');
        }

        summaryEl.classList.remove('d-none');

        const clusterChips = selectedClusters.map(cluster => {
            return `<li class="list-group-item px-2 py-1 d-flex justify-content-between">
                <span class="fw-semibold">${cluster.name}</span>
                <span class="text-muted small">${cluster.standards_count} standards</span>
            </li>`;
        }).join('');

        summaryEl.innerHTML = `
            <p class="text-muted">${count} clusters selected. Visualizations reflect the combined standards set.</p>
            <h6 class="mt-3">Included Clusters</h6>
            <ul class="list-group list-group-flush mb-3">${clusterChips}</ul>
        `;
    }

    function focusOnCluster(clusterId) {
        if (!clusterId) {
            return;
        }
        state.activeClusterId = clusterId;
        const cluster = state.selectedClusters.get(clusterId) || state.clusterDetails.get(clusterId);
        if (cluster) {
            renderClusterDetail(cluster);
        } else {
            ensureClusterDetail(clusterId).then(detail => {
                if (detail) {
                    state.clusterDetails.set(detail.id, detail);
                    renderClusterDetail(detail);
                }
            }).catch(err => console.error(err));
        }
    }

    function handleClusterSelection(clusterId, isSelected) {
        if (!clusterId) {
            return Promise.resolve(null);
        }

        if (isSelected) {
            return ensureClusterDetail(clusterId).then(cluster => {
                if (!cluster) {
                    return null;
                }
                state.selectedClusters.set(clusterId, cluster);
                state.activeClusterId = clusterId;
                if (state.selectedClusters.size === 1) {
                    renderClusterDetail(cluster);
                } else {
                    renderMultiClusterSummary(getSelectedClustersArray());
                }
                updateSelectionUI();
                refreshVisualizationForSelection();
                return cluster;
            }).catch(err => {
                console.error(err);
                return null;
            });
        }

        state.selectedClusters.delete(clusterId);
        if (state.activeClusterId === clusterId) {
            state.activeClusterId = state.selectedClusters.size ? getSelectedClustersArray()[0].id : null;
        }

        if (state.selectedClusters.size === 1) {
            const [onlyCluster] = getSelectedClustersArray();
            renderClusterDetail(onlyCluster);
        } else if (state.selectedClusters.size > 1) {
            renderMultiClusterSummary(getSelectedClustersArray());
        } else {
            renderClusterDetail(null);
        }

        updateSelectionUI();
        refreshVisualizationForSelection();
        return Promise.resolve(null);
    }

    function handleSaveCluster(selected) {
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

        const isEdit = state.modalMode === 'edit' && state.editingClusterId;
        const urlBase = isEdit ? state.endpoints.clusterDetailBase : state.endpoints.clustersEndpoint;
        const url = isEdit
            ? buildEndpoint(urlBase, `${state.editingClusterId}/`)
            : buildEndpoint(urlBase);
        if (!url) {
            console.error('Clusters endpoint not configured');
            return;
        }

        const method = isEdit ? 'PATCH' : 'POST';

        fetchJSON(url, {
            method,
            headers,
            body: JSON.stringify(payload)
        }).then(data => {
            if (state.modal) {
                state.modal.hide();
            }
            state.modalMode = 'create';
            state.editingClusterId = null;
            const cluster = data.data || data;
            if (cluster && cluster.id) {
                state.clusterDetails.set(cluster.id, cluster);
                loadClusterList();
                if (isEdit) {
                    if (state.selectedClusters.has(cluster.id)) {
                        state.selectedClusters.set(cluster.id, cluster);
                    }
                    if (state.activeClusterId === cluster.id) {
                        renderClusterDetail(cluster);
                    }
                    updateSelectionUI();
                    refreshVisualizationForSelection();
                } else {
                    loadClusterDetail(cluster.id);
                }
            }
        }).catch(err => {
            console.error(err);
            alert(err.message || (isEdit ? 'Failed to update cluster.' : 'Failed to create cluster.'));
        });
    }

    function openClusterModal({ mode = 'create', cluster = null } = {}) {
        const modalEl = document.getElementById('cluster-builder-modal');
        if (!modalEl || !window.bootstrap) {
            console.warn('Cluster builder modal or Bootstrap not available');
            return;
        }

        if (!state.modal) {
            state.modal = new window.bootstrap.Modal(modalEl);
        }
        state.modalMode = mode;
        state.editingClusterId = cluster && cluster.id ? cluster.id : null;

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
                onConfirmSelection: handleSaveCluster
            });
        }

        const titleInput = modalEl.querySelector('#cluster-title-input');
        const descriptionInput = modalEl.querySelector('#cluster-description-input');
        const confirmButton = modalEl.querySelector('[data-confirm-selection]');
        if (confirmButton) {
            confirmButton.textContent = mode === 'edit' ? 'Save Changes' : 'Save Cluster';
        }

        if (mode === 'edit' && cluster) {
            if (titleInput) titleInput.value = cluster.name || '';
            if (descriptionInput) descriptionInput.value = cluster.description || '';
            const preselected = (cluster.members || []).map(member => ({
                id: member.id,
                title: member.title,
                code: member.code,
                state: member.state,
                similarity_score: member.similarity_score
            }));
            if (state.searchInstance && typeof state.searchInstance.setSelection === 'function') {
                state.searchInstance.setSelection(preselected);
            }
            state.lastSearchPayload = cluster.search_context || {};
            if (state.searchInstance && typeof state.searchInstance.setFilters === 'function') {
                state.searchInstance.setFilters(state.lastSearchPayload || {});
            }
        } else {
            if (titleInput) titleInput.value = '';
            if (descriptionInput) descriptionInput.value = '';
            state.lastSearchPayload = null;
            if (state.searchInstance) {
                if (typeof state.searchInstance.clearSelection === 'function') {
                    state.searchInstance.clearSelection();
                }
                if (typeof state.searchInstance.setSelection === 'function') {
                    state.searchInstance.setSelection([]);
                }
                if (typeof state.searchInstance.setFilters === 'function') {
                    state.searchInstance.setFilters();
                }
            }
        }

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

            const gradeFilter = document.getElementById('cluster-grade-filter');
            const subjectFilter = document.getElementById('cluster-subject-filter');
            if (gradeFilter) {
                gradeFilter.addEventListener('change', handleFilterChange);
                state.activeFilters.gradeLevel = gradeFilter.value || '';
            }
            if (subjectFilter) {
                subjectFilter.addEventListener('change', handleFilterChange);
                state.activeFilters.subjectArea = subjectFilter.value || '';
            }
            const vizButtons = document.querySelectorAll('[data-viz-mode]');
            vizButtons.forEach(button => {
                button.addEventListener('click', event => {
                    event.preventDefault();
                    const mode = button.dataset.vizMode;
                    setVizMode(mode);
                });
            });
            syncHiddenFilters();
            updateVizModeButtons();

            const compareBtn = document.getElementById('compare-selected-btn');
            if (compareBtn) {
                compareBtn.addEventListener('click', event => {
                    event.preventDefault();
                    if (!state.selectedClusters.size) {
                        return;
                    }
                    if (state.selectedClusters.size === 1) {
                        const [onlyCluster] = getSelectedClustersArray();
                        renderClusterDetail(onlyCluster);
                    } else {
                        renderMultiClusterSummary(getSelectedClustersArray());
                    }
                    refreshVisualizationForSelection();
                });
            }

            const trigger = document.getElementById('new-cluster-btn');
            if (trigger) {
                trigger.addEventListener('click', () => openClusterModal({ mode: 'create' }));
            }
        },
        refresh() {
            loadClusterList();
            loadReportList();
        },
        openModal(options) {
            openClusterModal(options);
        },
        deleteCluster(clusterId) {
            return performClusterDeletion(clusterId);
        }
    };
})();
function handleVisualizationPanel(cluster) {
    const title = document.getElementById('cluster-detail-title');
    const placeholder = document.getElementById('cluster-placeholder');
    const summaryEl = document.getElementById('cluster-summary');
    if (!title || !summaryEl) return;

    title.textContent = cluster.name || 'Custom Cluster';
    if (placeholder) {
        placeholder.classList.add('d-none');
    }

    const members = (cluster.members || []).map(member => {
        return `<li><strong>${member.code}</strong> – ${member.title || 'Untitled'} <span class="text-muted">(${member.state || 'N/A'})</span></li>`;
    }).join('');
    const summary = cluster.coverage_summary || {};

    summaryEl.classList.remove('d-none');
    const coverageSections = [];
    if (summary.states && Object.keys(summary.states).length) {
        const stateItems = Object.entries(summary.states).map(([code, count]) => `<li><strong>${code}</strong>: ${count}</li>`).join('');
        coverageSections.push(`<div class="mb-2"><strong>States</strong><ul class="mb-0 ps-3 small">${stateItems}</ul></div>`);
    }
    if (summary.subjects && Object.keys(summary.subjects).length) {
        const subjectItems = Object.entries(summary.subjects).map(([name, count]) => `<li><strong>${name}</strong>: ${count}</li>`).join('');
        coverageSections.push(`<div class="mb-2"><strong>Subjects</strong><ul class="mb-0 ps-3 small">${subjectItems}</ul></div>`);
    }
    if (summary.grades && Object.keys(summary.grades).length) {
        const gradeItems = Object.entries(summary.grades).map(([grade, count]) => `<li><strong>${grade}</strong>: ${count}</li>`).join('');
        coverageSections.push(`<div class="mb-2"><strong>Grades</strong><ul class="mb-0 ps-3 small">${gradeItems}</ul></div>`);
    }

    const editButtonMarkup = cluster.can_edit ? `
        <div class="d-flex justify-content-end gap-2 mb-3">
            <button type="button" class="btn btn-sm btn-outline-primary" data-edit-cluster="${cluster.id}">
                <i class="fas fa-edit"></i> Edit Cluster
            </button>
            <button type="button" class="btn btn-sm btn-outline-danger" data-delete-cluster="${cluster.id}">
                <i class="fas fa-trash-alt"></i> Delete Cluster
            </button>
        </div>
    ` : '';

    summaryEl.innerHTML = `
        ${editButtonMarkup}
        <p class="text-muted">${cluster.description || 'No description provided.'}</p>
        <h6>Members (${cluster.standards_count})</h6>
        <ul class="ps-3">${members || '<li>No standards yet.</li>'}</ul>
        ${coverageSections.length ? `<h6 class="mt-3">Coverage Snapshot</h6>${coverageSections.join('')}` : ''}
    `;

    if (cluster.can_edit) {
        const editButton = summaryEl.querySelector('[data-edit-cluster]');
        if (editButton) {
            editButton.addEventListener('click', event => {
                event.preventDefault();
                if (window.dashboardCustomClusters && typeof window.dashboardCustomClusters.openModal === 'function') {
                    window.dashboardCustomClusters.openModal({ mode: 'edit', cluster });
                }
            });
        }

        const deleteButton = summaryEl.querySelector('[data-delete-cluster]');
        if (deleteButton) {
            deleteButton.addEventListener('click', event => {
                event.preventDefault();
                const confirmed = window.confirm('Delete this custom cluster? This action cannot be undone.');
                if (!confirmed) {
                    return;
                }

                const target = event.currentTarget;
                const originalHtml = target.innerHTML;
                target.disabled = true;
                target.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>Deleting...';

                if (window.dashboardCustomClusters && typeof window.dashboardCustomClusters.deleteCluster === 'function') {
                    window.dashboardCustomClusters.deleteCluster(cluster.id)
                        .catch(err => {
                            target.disabled = false;
                            target.innerHTML = originalHtml;
                            const message = (err && err.message) ? err.message : 'Unable to delete cluster.';
                            window.alert(message);
                        });
                }
            });
        }
    }
}
