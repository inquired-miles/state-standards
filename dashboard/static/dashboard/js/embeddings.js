(function () {
    if (window.dashboardEmbeddings) {
        return;
    }

    let standardIdFilter = [];
    let manualClusterMap = {};
    let autoInit = true;
    let endpoints = {};

    function setStandardFilter(ids) {
        standardIdFilter = Array.isArray(ids) ? ids : [];
    }

    function buildParams(formData = {}) {
        const params = new URLSearchParams();
        Object.entries(formData).forEach(([key, value]) => {
            if (value !== undefined && value !== null && value !== '') {
                params.append(key, value);
            }
        });
        standardIdFilter.forEach(id => params.append('standard_ids', id));
        if (manualClusterMap && Object.keys(manualClusterMap).length) {
            params.append('cluster_labels', JSON.stringify(manualClusterMap));
        }
        return params;
    }

    let networkData = null;

    window.dashboardEmbeddings = {
        setStandardFilter,
        setManualClusterMap(map) {
            manualClusterMap = map && typeof map === 'object' ? JSON.parse(JSON.stringify(map)) : {};
        },
        getManualClusterMap() {
            return JSON.parse(JSON.stringify(manualClusterMap || {}));
        },
        buildParams,
        getStandardFilter: () => [...standardIdFilter],
        setAutoInit(value) {
            autoInit = Boolean(value);
        },
        shouldAutoInit() {
            return autoInit;
        },
        setEndpoints(newEndpoints) {
            endpoints = Object.assign({}, endpoints, newEndpoints);
        },
        getEndpoint(key) {
            return endpoints[key];
        },
        getEndpoints() {
            return Object.assign({}, endpoints);
        },
        setNetworkData(data) {
            networkData = data;
        },
        getNetworkData() {
            return networkData ? JSON.parse(JSON.stringify(networkData)) : null;
        }
    };
})();
