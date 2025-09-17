(function () {
    if (window.dashboardSemanticSearch) {
        return;
    }

    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) {
            return parts.pop().split(';').shift();
        }
        return null;
    }

    function toggle(element, show) {
        if (!element) return;
        element.classList.toggle('d-none', !show);
    }

    function renderStateEmphasis(container, data, onFilter) {
        container.innerHTML = '';
        Object.entries(data).forEach(([state, stateData]) => {
            const item = document.createElement('div');
            item.className = 'state-emphasis-item clickable';
            item.style.cursor = 'pointer';
            item.innerHTML = `
                <div>
                    <strong>${state}</strong>
                    <br><small>${stateData.count} matches</small>
                </div>
            `;
            item.addEventListener('click', () => onFilter(state));
            container.appendChild(item);
        });
    }

    function renderResultItem(result, options) {
        const container = document.createElement('div');
        container.className = 'search-result-item';

        const alignmentClass = {
            aligned: 'bg-success',
            could_align: 'bg-warning',
            stretch: 'bg-info',
            not_aligned: 'bg-secondary'
        }[result.alignment_category] || 'bg-secondary';

        const checkbox = options.mode === 'select'
            ? `<div class="form-check me-3">
                    <input class="form-check-input" type="checkbox" value="${result.id}" data-result-select>
                </div>`
            : '';

        container.innerHTML = `
            <div class="d-flex justify-content-between align-items-start">
                <div class="d-flex">
                    ${checkbox}
                    <div>
                        <strong>${result.title}</strong>
                        <span class="alignment-label ms-2 badge ${alignmentClass}">${result.alignment_label}</span>
                        <p class="mb-1 mt-1 text-muted small">${result.description || ''}</p>
                        <small><strong>${result.state}</strong> • ${result.code}</small>
                    </div>
                </div>
            </div>
        `;

        if (options.mode === 'select') {
            const input = container.querySelector('[data-result-select]');
            input.addEventListener('change', () => {
                options.onToggle(result, input.checked);
            });
        }

        return container;
    }

    function renderSelectedList(container, countEl, selected) {
        if (!container || !countEl) return;
        container.innerHTML = '';
        const entries = Array.from(selected.values());
        countEl.textContent = entries.length;

        if (!entries.length) {
            container.innerHTML = '<div class="text-muted">No standards selected yet.</div>';
            return;
        }

        entries.forEach(item => {
            const row = document.createElement('div');
            row.className = 'd-flex justify-content-between align-items-center border-bottom py-1';
            row.innerHTML = `
                <div>
                    <strong>${item.title}</strong>
                    <br><small>${item.state} • ${item.code}</small>
                </div>
                <button type="button" class="btn btn-sm btn-outline-danger" data-remove="${item.id}">Remove</button>
            `;
            row.querySelector('[data-remove]').addEventListener('click', () => {
                selected.delete(item.id);
                const checkbox = container.ownerDocument.querySelector(`[data-result-select][value="${item.id}"]`);
                if (checkbox) {
                    checkbox.checked = false;
                }
                renderSelectedList(container, countEl, selected);
            });
            container.appendChild(row);
        });
    }

    window.dashboardSemanticSearch = {
        init(options) {
            const root = options.root;
            if (!root) {
                console.warn('Semantic search root not found');
                return;
            }

            const queryInput = root.querySelector('[data-search-query]');
            const searchButton = root.querySelector('[data-search-button]');
            const limitSelect = root.querySelector('[data-search-limit]');
            const resultsContainer = root.querySelector('[data-results-container]');
            const resultsList = root.querySelector('[data-results-list]');
            const stateContainer = root.querySelector('[data-state-emphasis]');
            const loadingEl = root.querySelector('[data-search-loading]');
            const selectionPanel = root.querySelector('[data-selection-panel]');
            const selectedList = root.querySelector('[data-selected-list]');
            const selectedCount = root.querySelector('[data-selection-count]');
            const clearButton = root.querySelector('[data-clear-selection]');
            const confirmButton = root.querySelector('[data-confirm-selection]');
            const subjectFilter = root.querySelector('[data-filter-subject]');
            const gradeFilter = root.querySelector('[data-filter-grade]');
            const stateFilter = root.querySelector('[data-filter-state]');

            const mode = (root.dataset.mode || 'view');
            const selected = new Map();
            let currentResults = [];
            let currentStateEmphasis = {};

            const csrfToken = options.csrfToken || getCookie('csrftoken');

            function collectFilters() {
                const filters = {};

                if (subjectFilter && subjectFilter.value) {
                    filters.subject_area = subjectFilter.value;
                }
                if (gradeFilter && gradeFilter.value) {
                    filters.grade_level = gradeFilter.value;
                }
                if (stateFilter && stateFilter.value) {
                    filters.state = stateFilter.value;
                }

                if (typeof options.getFilters === 'function') {
                    Object.assign(filters, options.getFilters());
                }
                return filters;
            }

            function updateSelectionUI() {
                if (selectedList && selectedCount) {
                    renderSelectedList(selectedList, selectedCount, selected);
                }
                if (resultsList) {
                    const selectedIds = new Set(Array.from(selected.keys()).map(String));
                    resultsList.querySelectorAll('[data-result-select]').forEach(cb => {
                        cb.checked = selectedIds.has(String(cb.value));
                    });
                }
            }

            function emitSelectionChange() {
                if (typeof options.onSelectionChange === 'function') {
                    options.onSelectionChange(Array.from(selected.values()));
                }
            }

            function clearSelection() {
                selected.clear();
                updateSelectionUI();
                emitSelectionChange();
            }

            function setSelection(items) {
                selected.clear();
                (items || []).forEach(item => {
                    if (item && item.id) {
                        selected.set(item.id, item);
                    }
                });
                updateSelectionUI();
                emitSelectionChange();
            }

            function handleToggle(result, checked) {
                if (checked) {
                    selected.set(result.id, result);
                } else {
                    selected.delete(result.id);
                }
                updateSelectionUI();
                emitSelectionChange();
            }

            function filterByState(state) {
                if (!resultsList) return;
                resultsList.innerHTML = '';
                const filtered = currentResults.filter(item => item.state === state);
                const header = document.createElement('div');
                header.className = 'alert alert-info d-flex justify-content-between align-items-center';
                header.innerHTML = `
                    <span><strong>Filtered by:</strong> ${state} (${filtered.length} results)</span>
                    <button class="btn btn-sm btn-outline-secondary" data-clear-state>Clear Filter</button>
                `;
                header.querySelector('[data-clear-state]').addEventListener('click', () => {
                    renderResults(currentResults, currentStateEmphasis);
                });
                resultsList.appendChild(header);
                filtered.forEach(result => {
                    resultsList.appendChild(renderResultItem(result, {
                        mode,
                        onToggle: handleToggle
                    }));
                });
            }

            function renderResults(results, stateData = {}) {
                currentResults = results;
                currentStateEmphasis = stateData;
                resultsList.innerHTML = '';
                results.forEach(result => {
                    const item = renderResultItem(result, {
                        mode,
                        onToggle: handleToggle
                    });
                    if (mode === 'select') {
                        const checkbox = item.querySelector('[data-result-select]');
                        if (checkbox && selected.has(result.id)) {
                            checkbox.checked = true;
                        }
                    }
                    resultsList.appendChild(item);
                });
                if (stateContainer) {
                    renderStateEmphasis(stateContainer, currentStateEmphasis, filterByState);
                }
                toggle(resultsContainer, true);
            }

            function showStateEmphasis(data) {
                if (!stateContainer) return;
                renderStateEmphasis(stateContainer, data, filterByState);
            }

            async function performSearch() {
                const query = queryInput.value.trim();
                if (!query) return;

                const payload = Object.assign({
                    query,
                    limit: parseInt(limitSelect.value, 10)
                }, collectFilters());

                toggle(loadingEl, true);
                toggle(resultsContainer, false);
                try {
                    const response = await fetch(options.searchUrl, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-CSRFToken': csrfToken
                        },
                        body: JSON.stringify(payload)
                    });
                    const json = await response.json();
                    if (!response.ok) {
                        throw new Error(json.error || 'Search failed');
                    }
                    const data = json.data || json;
                    renderResults(data.results || [], data.state_emphasis || {});
                    showStateEmphasis(data.state_emphasis || {});
                    if (typeof options.onSearchComplete === 'function') {
                        options.onSearchComplete(data, payload);
                    }
                } catch (error) {
                    console.error('Semantic search failed:', error);
                    resultsList.innerHTML = `<div class="alert alert-danger">${error.message}</div>`;
                    toggle(resultsContainer, true);
                } finally {
                    toggle(loadingEl, false);
                }
            }

            if (mode === 'select') {
                toggle(selectionPanel, true);
                updateSelectionUI();
                if (clearButton) {
                    clearButton.addEventListener('click', clearSelection);
                }
                if (confirmButton) {
                    confirmButton.addEventListener('click', () => {
                        if (typeof options.onConfirmSelection === 'function') {
                            options.onConfirmSelection(Array.from(selected.values()));
                        }
                    });
                }
            } else {
                toggle(selectionPanel, false);
            }

            searchButton.addEventListener('click', performSearch);
            queryInput.addEventListener('keypress', event => {
                if (event.key === 'Enter') {
                    performSearch();
                }
            });

            const filterElements = [subjectFilter, gradeFilter, stateFilter].filter(Boolean);
            if (filterElements.length) {
                const handleFilterChange = () => {
                    if (queryInput.value.trim()) {
                        performSearch();
                    }
                };
                filterElements.forEach(element => {
                    element.addEventListener('change', handleFilterChange);
                });
            }

            return {
                search: performSearch,
                clearSelection,
                getSelected: () => Array.from(selected.values()),
                setSelection,
                setFilters(filterValues = {}) {
                    if (subjectFilter) {
                        subjectFilter.value = filterValues.subject_area || '';
                    }
                    if (gradeFilter) {
                        gradeFilter.value = filterValues.grade_level || '';
                    }
                    if (stateFilter) {
                        stateFilter.value = filterValues.state || '';
                    }
                }
            };
        }
    };
})();
