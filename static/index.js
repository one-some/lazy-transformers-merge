function backendError(text) {
    alert(text);
}

function renderParam(param, parent) {
    return $e(
        "model-parameter",
        parent,
        { innerText: param.name }
    );
}

function renderModel(container, model) {
    let doneIndexes = [];
    for (const param of model.parameters) {
        // TODO: Render individual params?

        if (param.layerIndex === null) continue;
        if (doneIndexes.includes(param.layerIndex)) continue;
        doneIndexes.push(param.layerIndex);

        const layerEl = $e("model-layer", container, {"layer-index": param.layerIndex});
        $e("span", layerEl, {
            classes: ["layer-label"],
            // innerText: `Layer ${param.layerIndex}`
            innerText: param.name,
        });

        // renderParam(param, layerEl || container);
    }
}

async function fetchAndRenderModels() {
    const r = await fetch("/api/models_info.json");
    if (!r.ok) return backendError(`Got HTTP ${r.status} for ${r.url}.`);

    const j = await r.json();

    for (const model of j) {
        const modelCont = $e("model-container", $el("#mixing-zone"));
        renderModel(modelCont, model);
    }
}

// Init;
fetchAndRenderModels();