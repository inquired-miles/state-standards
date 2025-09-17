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
        activeReportId: null,
        activeFilters: {
            gradeLevel: '',
            subjectArea: ''
        },
        vizMode: '2d',
        modalMode: 'create',
        editingClusterId: null,
        reportModal: null,
        reportModalMode: 'create',
        editingReportId: null,
        selectionRequestId: 0,
        activeReport: null
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
            listItem.className = 'list-group-item list-group-item-action d-flex align-items-center justify-content-between gap-2';

            const leftWrap = document.createElement('div');
            leftWrap.className = 'd-flex align-items-center flex-grow-1 gap-2';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.className = 'form-check-input position-static';
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

            leftWrap.appendChild(checkbox);
            leftWrap.appendChild(label);

            const actions = document.createElement('div');
            actions.className = 'btn-group btn-group-sm';

            const editButton = document.createElement('button');
            editButton.type = 'button';
            editButton.className = 'btn btn-outline-primary';
            editButton.textContent = 'Edit';
            editButton.addEventListener('click', event => {
                event.preventDefault();
                event.stopPropagation();
                ensureClusterDetail(cluster.id).then(detail => {
                    if (detail && window.dashboardCustomClusters && typeof window.dashboardCustomClusters.openModal === 'function') {
                        window.dashboardCustomClusters.openModal({ mode: 'edit', cluster: detail });
                    }
                });
            });

            const deleteButton = document.createElement('button');
            deleteButton.type = 'button';
            deleteButton.className = 'btn btn-outline-danger';
            deleteButton.textContent = 'Delete';
            deleteButton.addEventListener('click', event => {
                event.preventDefault();
                event.stopPropagation();
                const confirmed = window.confirm('Delete this custom cluster? This action cannot be undone.');
                if (!confirmed) {
                    return;
                }
                if (window.dashboardCustomClusters && typeof window.dashboardCustomClusters.deleteCluster === 'function') {
                    window.dashboardCustomClusters.deleteCluster(cluster.id)
                        .catch(err => {
                            const message = (err && err.message) ? err.message : 'Unable to delete cluster.';
                            window.alert(message);
                        });
                }
            });

            actions.appendChild(editButton);
            actions.appendChild(deleteButton);

            checkbox.addEventListener('change', event => {
                handleClusterSelection(cluster.id, event.target.checked);
            });

            listItem.addEventListener('click', event => {
                if (event.target === checkbox || event.target.closest('.btn-group')) {
                    return;
                }
                event.preventDefault();
                checkbox.checked = !checkbox.checked;
                handleClusterSelection(cluster.id, checkbox.checked);
            });

            listItem.appendChild(leftWrap);
            listItem.appendChild(actions);
            container.appendChild(listItem);
        });

        updateSelectionUI();
        syncClusterCheckboxes();
    }

    function renderClusterDetail(cluster) {
        const placeholder = document.getElementById('cluster-placeholder');
        const vizContainer = document.getElementById('cluster-visualizations');

        if (!placeholder || !vizContainer) {
            return;
        }

        if (!cluster) {
            placeholder.classList.remove('d-none');
            placeholder.textContent = 'Choose one or more clusters from the list to explore their coverage and relationships.';
            vizContainer.classList.add('d-none');
            return;
        }

        placeholder.classList.add('d-none');
        vizContainer.classList.remove('d-none');
        handleVisualizationPanel();
    }

    function renderReportList(payload) {
        const container = document.getElementById('report-list');
        if (!container) return;
        container.innerHTML = '';
        const reports = (payload && payload.data && payload.data.reports) || [];
        if (!reports.length) {
            container.innerHTML = '<div class="text-muted small">No coverage reports yet.</div>';
            state.activeReportId = null;
            state.activeReport = null;
            updateReportSummary(null);
            setSelectedClustersFromIds([]);
            return;
        }
        reports.forEach(report => {
            const listItem = document.createElement('div');
            listItem.className = 'list-group-item list-group-item-action d-flex align-items-center justify-content-between gap-2';
            listItem.dataset.reportId = report.id;
            if (state.activeReportId === report.id) {
                listItem.classList.add('active');
            }

            const leftWrap = document.createElement('div');
            leftWrap.className = 'flex-grow-1';
            leftWrap.innerHTML = `
                <div class="fw-semibold">${report.title}</div>
                ${report.description ? `<div class="text-muted small">${report.description}</div>` : ''}
            `;

            const actions = document.createElement('div');
            actions.className = 'btn-group btn-group-sm';

            const editButton = document.createElement('button');
            editButton.type = 'button';
            editButton.className = 'btn btn-outline-primary';
            editButton.textContent = 'Edit';
            editButton.addEventListener('click', event => {
                event.preventDefault();
                event.stopPropagation();
                openCoverageReportModal({ mode: 'edit', report });
            });

            const deleteButton = document.createElement('button');
            deleteButton.type = 'button';
            deleteButton.className = 'btn btn-outline-danger';
            deleteButton.textContent = 'Delete';
            deleteButton.addEventListener('click', event => {
                event.preventDefault();
                event.stopPropagation();
                handleDeleteReport(report.id);
            });

            actions.appendChild(editButton);
            actions.appendChild(deleteButton);

            listItem.addEventListener('click', event => {
                if (event.target.closest('.btn-group')) {
                    return;
                }
                state.activeReportId = report.id;
                container.querySelectorAll('.list-group-item').forEach(btn => btn.classList.remove('active'));
                listItem.classList.add('active');
                loadReportDetail(report.id);
            });

            listItem.appendChild(leftWrap);
            listItem.appendChild(actions);
            container.appendChild(listItem);
        });

        const activeReportStillExists = reports.some(report => report.id === state.activeReportId);
        if (!activeReportStillExists) {
            state.activeReportId = reports[0].id;
        }

        if (state.activeReportId) {
            const activeButton = container.querySelector(`[data-report-id="${state.activeReportId}"]`);
            if (activeButton) {
                activeButton.classList.add('active');
            }
            loadReportDetail(state.activeReportId);
        }
    }

    function updateReportSummary(report) {
        const summaryCard = document.getElementById('report-summary');
        const titleEl = document.getElementById('report-summary-title');
        const descriptionEl = document.getElementById('report-summary-description');
        const saveButton = document.getElementById('save-report-changes-btn');

        if (!summaryCard || !titleEl || !descriptionEl) {
            return;
        }

        if (!report) {
            summaryCard.hidden = true;
            titleEl.textContent = '';
            descriptionEl.textContent = '';
            descriptionEl.hidden = true;
            if (saveButton) saveButton.hidden = true;
            return;
        }

        summaryCard.hidden = false;
        titleEl.textContent = report.title || 'Coverage Report';
        if (report.description) {
            descriptionEl.hidden = false;
            descriptionEl.textContent = report.description;
        } else {
            descriptionEl.hidden = true;
            descriptionEl.textContent = '';
        }

        if (saveButton) saveButton.hidden = false;
    }

    function applyReportData(payload) {
        const report = payload && (payload.data || payload);
        if (!report) {
            updateReportSummary(null);
            return;
        }

        state.activeReport = report;
        const reportId = report.id || report.report_id || state.activeReportId;
        if (reportId) {
            state.activeReportId = reportId;
        }

        updateReportSummary(report);

        const entries = report.clusters || [];
        const clusterIds = entries.map(entry => entry.cluster_id).filter(Boolean);

        setSelectedClustersFromIds(clusterIds).catch(err => {
            console.error('Failed to apply coverage report selection', err);
        });
    }

    function setSelectedClustersFromIds(clusterIds) {
        const uniqueIds = Array.from(new Set(clusterIds.filter(Boolean)));

        const currentRequestId = ++state.selectionRequestId;

        if (!uniqueIds.length) {
            state.selectedClusters.clear();
            state.activeClusterId = null;
            updateSelectionUI();
            renderClusterDetail(null);
            refreshVisualizationForSelection();
            syncClusterCheckboxes();
            return Promise.resolve([]);
        }

        const detailPromises = uniqueIds.map(id => ensureClusterDetail(id));
        return Promise.all(detailPromises).then(clusters => {
            if (currentRequestId !== state.selectionRequestId) {
                return [];
            }
            state.selectedClusters.clear();
            clusters.forEach(cluster => {
                if (cluster && cluster.id) {
                    state.selectedClusters.set(cluster.id, cluster);
                }
            });

            const selectedArray = getSelectedClustersArray();
            if (selectedArray.length === 1) {
                state.activeClusterId = selectedArray[0].id;
                renderClusterDetail(selectedArray[0]);
            } else if (selectedArray.length > 1) {
                state.activeClusterId = selectedArray[0]?.id || null;
                renderMultiClusterSummary(selectedArray);
            } else {
                state.activeClusterId = null;
                renderClusterDetail(null);
            }

            updateSelectionUI();
            refreshVisualizationForSelection();
            syncClusterCheckboxes();
            return selectedArray;
        });
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
            .then(applyReportData)
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
        // Selection count no longer displayed; keep hook for potential future use.
    }

    function syncClusterCheckboxes() {
        const checkboxes = document.querySelectorAll('#cluster-list input[type="checkbox"][data-cluster-id]');
        const selectedIds = new Set(state.selectedClusters.keys());
        checkboxes.forEach(cb => {
            const clusterId = cb.dataset.clusterId;
            cb.checked = selectedIds.has(clusterId);
        });
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
        const placeholder = document.getElementById('cluster-placeholder');

        const count = selectedClusters.length;
        if (!placeholder) {
            return;
        }

        const vizContainer = document.getElementById('cluster-visualizations');
        if (vizContainer) {
            vizContainer.classList.remove('d-none');
        }

        const clusterItems = selectedClusters.map(cluster => {
            return `<li class="list-group-item px-2 py-1 d-flex justify-content-between">
                <span class="fw-semibold">${cluster.name}</span>
                <span class="text-muted small">${cluster.standards_count} standards</span>
            </li>`;
        }).join('');

        placeholder.classList.remove('d-none');
        placeholder.innerHTML = `
            <p class="text-muted mb-2">${count} clusters selected. Visualizations reflect the combined standards set.</p>
            <ul class="list-group list-group-flush mb-0">${clusterItems}</ul>
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
        syncClusterCheckboxes();
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

    function openCoverageReportModal({ mode = 'create', report = null } = {}) {
        const modalEl = document.getElementById('report-builder-modal');
        if (!modalEl || !window.bootstrap) {
            console.warn('Report builder modal or Bootstrap not available');
            return;
        }

        if (!state.reportModal) {
            state.reportModal = new window.bootstrap.Modal(modalEl);
        }

        state.reportModalMode = mode;
        state.editingReportId = report && report.id ? report.id : null;

        const modalTitle = modalEl.querySelector('#report-builder-label');
        const saveButton = modalEl.querySelector('#save-report-btn');
        if (modalTitle) {
            modalTitle.textContent = mode === 'edit' ? 'Edit Coverage Report' : 'New Coverage Report';
        }
        if (saveButton) {
            saveButton.innerHTML = mode === 'edit'
                ? '<i class="fas fa-save"></i> Save Changes'
                : '<i class="fas fa-save"></i> Save Coverage Report';
            saveButton.disabled = false;
        }

        const titleInput = modalEl.querySelector('#report-title-input');
        const descriptionInput = modalEl.querySelector('#report-description-input');
        if (mode === 'edit' && report) {
            if (titleInput) {
                titleInput.value = report.title || '';
            }
            if (descriptionInput) {
                descriptionInput.value = report.description || '';
            }
        } else {
            if (titleInput) {
                titleInput.value = '';
            }
            if (descriptionInput) {
                descriptionInput.value = '';
            }
        }

        state.reportModal.show();
    }

    function handleSaveReport() {
        const modalEl = document.getElementById('report-builder-modal');
        if (!modalEl) {
            console.warn('Report builder modal missing');
            return;
        }

        const title = (modalEl.querySelector('#report-title-input')?.value || '').trim();
        const description = (modalEl.querySelector('#report-description-input')?.value || '').trim();
        const clusterIds = Array.from(state.selectedClusters.keys());
        if (!clusterIds.length) {
            alert('Select at least one cluster before saving the coverage report.');
            return;
        }

        if (!title) {
            alert('Please provide a title for the coverage report.');
            return;
        }

        const payload = {
            title,
            description,
            cluster_ids: clusterIds,
            notes: {}
        };

        const headers = {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest',
            'X-CSRFToken': state.csrfToken || getCookie('csrftoken')
        };

        const url = buildEndpoint(state.endpoints.reportsEndpoint);
        if (!url) {
            console.error('Reports endpoint not configured');
            return;
        }

        fetchJSON(url, {
            method: 'POST',
            headers,
            body: JSON.stringify(payload)
        }).then(data => {
            if (state.reportModal) {
                state.reportModal.hide();
            }
            state.reportModalMode = 'create';
            state.editingReportId = null;

            const report = data.data || data;
            loadReportList();

            if (report && report.id) {
                state.activeReportId = report.id;
                loadReportDetail(report.id);
            }
        }).catch(error => {
            console.error('Failed to save coverage report', error);
            alert(error.message || 'Failed to save coverage report.');
        });
    }

    function handleSaveReportChanges() {
        if (!state.activeReportId) {
            alert('Select a coverage report before saving changes.');
            return;
        }

        const clusterIds = Array.from(state.selectedClusters.keys());
        if (!clusterIds.length) {
            alert('Select at least one cluster before saving the coverage report.');
            return;
        }

        const url = buildEndpoint(state.endpoints.reportDetailBase, `${state.activeReportId}/`);
        if (!url) {
            console.error('Report detail endpoint not configured');
            return;
        }

        const headers = {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest',
            'X-CSRFToken': state.csrfToken || getCookie('csrftoken')
        };

        const payload = {
            cluster_ids: clusterIds
        };

        fetchJSON(url, {
            method: 'PATCH',
            headers,
            body: JSON.stringify(payload)
        }).then(data => {
            applyReportData(data);
            loadReportList();
        }).catch(error => {
            console.error('Failed to update coverage report', error);
            alert(error.message || 'Failed to update coverage report.');
        });
    }

    function handleSaveReportMetadata() {
        if (!state.editingReportId) {
            console.warn('No report selected for editing');
            return;
        }

        const modalEl = document.getElementById('report-builder-modal');
        if (!modalEl) {
            return;
        }

        const title = (modalEl.querySelector('#report-title-input')?.value || '').trim();
        const description = (modalEl.querySelector('#report-description-input')?.value || '').trim();

        if (!title) {
            alert('Please provide a title for the coverage report.');
            return;
        }

        const url = buildEndpoint(state.endpoints.reportDetailBase, `${state.editingReportId}/`);
        if (!url) {
            console.error('Report detail endpoint not configured');
            return;
        }

        const headers = {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest',
            'X-CSRFToken': state.csrfToken || getCookie('csrftoken')
        };

        const payload = {
            title,
            description
        };

        fetchJSON(url, {
            method: 'PATCH',
            headers,
            body: JSON.stringify(payload)
        }).then(data => {
            if (state.reportModal) {
                state.reportModal.hide();
            }
            state.reportModalMode = 'create';
            state.editingReportId = null;
            applyReportData(data);
            loadReportList();
        }).catch(error => {
            console.error('Failed to update coverage report metadata', error);
            alert(error.message || 'Failed to update coverage report.');
        });
    }

    function handleDeleteReport(reportId) {
        const targetId = reportId || state.activeReportId;
        if (!targetId) {
            return;
        }

        if (!confirm('Delete this coverage report? This action cannot be undone.')) {
            return;
        }

        const url = buildEndpoint(state.endpoints.reportDetailBase, `${targetId}/`);
        if (!url) {
            console.error('Report detail endpoint not configured');
            return;
        }

        fetchJSON(url, {
            method: 'DELETE',
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'X-CSRFToken': state.csrfToken || getCookie('csrftoken')
            }
        }).then(() => {
            if (targetId === state.activeReportId) {
                state.activeReportId = null;
                state.activeReport = null;
            }
            updateReportSummary(null);
            setSelectedClustersFromIds([]);
            loadReportList();
        }).catch(error => {
            console.error('Failed to delete coverage report', error);
            alert(error.message || 'Failed to delete coverage report.');
        });
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

            const trigger = document.getElementById('new-cluster-btn');
            if (trigger) {
                trigger.addEventListener('click', () => openClusterModal({ mode: 'create' }));
            }

            const newReportButton = document.getElementById('new-report-btn');
            if (newReportButton) {
                newReportButton.addEventListener('click', event => {
                    event.preventDefault();
                    openCoverageReportModal({ mode: 'create' });
                });
            }

            const saveReportButton = document.getElementById('save-report-btn');
            if (saveReportButton) {
                saveReportButton.addEventListener('click', event => {
                    event.preventDefault();
                    if (state.reportModalMode === 'edit') {
                        handleSaveReportMetadata();
                    } else {
                        handleSaveReport();
                    }
                });
            }

            const saveReportChangesButton = document.getElementById('save-report-changes-btn');
            if (saveReportChangesButton) {
                saveReportChangesButton.addEventListener('click', event => {
                    event.preventDefault();
                    handleSaveReportChanges();
                });
            }

        },
        refresh() {
            loadClusterList();
            loadReportList();
        },
        openModal(options) {
            openClusterModal(options);
        },
        openReportModal(options) {
            openCoverageReportModal(options || {});
        },
        deleteCluster(clusterId) {
            return performClusterDeletion(clusterId);
        }
    };
})();
function handleVisualizationPanel() {
    const placeholder = document.getElementById('cluster-placeholder');
    const vizContainer = document.getElementById('cluster-visualizations');
    if (placeholder) {
        placeholder.classList.add('d-none');
    }
    if (vizContainer) {
        vizContainer.classList.remove('d-none');
    }
}
