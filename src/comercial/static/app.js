const DIAS = [
  { id: 1, lbl: "L" },
  { id: 2, lbl: "M" },
  { id: 3, lbl: "X" },
  { id: 4, lbl: "J" },
  { id: 5, lbl: "V" },
  { id: 6, lbl: "S" },
  { id: 7, lbl: "D" },
];

const SALUDO_ONBOARDING =
  "Hola. Cuéntame cómo se llama el negocio, qué ofrece, sus precios y horarios (o pega directamente su lista de servicios) y yo relleno el formulario por ti.";

const state = {
  token: localStorage.getItem("sma_token") || "",
  rol: localStorage.getItem("sma_rol") || "",
  negocioId: localStorage.getItem("sma_negocio_id") || "",
  rolLogin: "dueno",
  tabActual: "alertas",
  resumen: null,
  diasSeleccionados: [1, 2, 3, 4, 5, 6],
  onbHistorial: [],
  onbBorrador: null,
  pollTimer: null,
};

const $ = (sel) => document.querySelector(sel);

function esc(valor) {
  return String(valor ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatoMoneda(valor, moneda = "COP") {
  const num = Number(valor || 0);
  return `$${num.toLocaleString("es-CO", { maximumFractionDigits: 0 })} ${esc(moneda)}`;
}

function formatearFechaLocal(isoStr, zonaHoraria = "America/Bogota") {
  if (!isoStr) return "";
  const dt = new Date(isoStr);
  if (Number.isNaN(dt.getTime())) {
    return String(isoStr).replace("T", " ").slice(0, 16);
  }
  try {
    const partes = new Intl.DateTimeFormat("sv-SE", {
      timeZone: zonaHoraria || "America/Bogota",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(dt);
    return partes.replace("T", " ").slice(0, 16);
  } catch {
    return dt.toLocaleString("es-CO", { hour12: false }).slice(0, 16);
  }
}

function mostrarToast(msg, tipo = "") {
  const el = $("#toast");
  el.textContent = msg;
  el.className = `toast ${tipo}`.trim();
  clearTimeout(mostrarToast._t);
  mostrarToast._t = setTimeout(() => el.classList.add("hidden"), 3500);
}

async function api(ruta, opts = {}) {
  const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  if (state.token) {
    headers.Authorization = `Bearer ${state.token}`;
  }
  const resp = await fetch(ruta, { ...opts, headers });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    if (resp.status === 401 && state.token) {
      cerrarSesion();
    }
    throw new Error(data.detail || `Error HTTP ${resp.status}`);
  }
  return data;
}

function cerrarSesion() {
  state.token = "";
  state.rol = "";
  state.negocioId = "";
  localStorage.removeItem("sma_token");
  localStorage.removeItem("sma_rol");
  localStorage.removeItem("sma_negocio_id");
  clearInterval(state.pollTimer);
  mostrarVista("login");
}

function mostrarVista(nombre) {
  $("#vista-login").classList.toggle("hidden", nombre !== "login");
  $("#vista-hub").classList.toggle("hidden", nombre !== "hub");
  $("#vista-negocio").classList.toggle("hidden", nombre !== "negocio");
  $("#btn-cerrar-sesion").classList.toggle("hidden", nombre === "login");
  $("#btn-volver-hub").classList.toggle("hidden", !(nombre === "negocio" && state.rol === "superadmin"));
}

async function cargarNegociosPublicos() {
  try {
    const data = await api("/api/public/negocios-acceso");
    const sel = $("#select-negocio-login");
    sel.innerHTML = (data.negocios || [])
      .map((n) => `<option value="${esc(n.id)}">${esc(n.nombre_negocio)} — ${esc(n.rubro)}</option>`)
      .join("");
    const params = new URLSearchParams(window.location.search);
    if (params.get("negocio")) {
      sel.value = params.get("negocio");
    }
  } catch (err) {
    mostrarToast(err.message, "error");
  }
}

async function cargarHub() {
  mostrarVista("hub");
  $("#topbar-title").textContent = "Centro Comercial IA";
  $("#topbar-sub").textContent = "Super-Admin";
  $("#hub-fecha").textContent = new Date().toLocaleDateString("es-CO", {
    weekday: "long",
    day: "numeric",
    month: "long",
  });
  try {
    const data = await api("/api/admin/negocios");
    $("#hub-motor").textContent = `${data.motor_ia} · ${data.gemini_activo ? "Activo" : "Sin API Key"}`;
    const grid = $("#hub-negocios-grid");
    if (!data.negocios || data.negocios.length === 0) {
      grid.innerHTML = `<div class="empty-state"><h3>Sin negocios registrados</h3><p>Crea el primero con el botón + Nuevo Negocio.</p></div>`;
      return;
    }
    grid.innerHTML = data.negocios
      .map((n) => {
        const m = n.metricas || {};
        const pausados = Number(m.chats_pausados || 0);
        const servs = Number(m.servicios_activos || 0);
        const citas = Number(m.citas_activas || 0);
        const badgePausados =
          pausados > 0
            ? `<span class="pill danger">${pausados} pausado(s)</span>`
            : `<span class="pill success">Al día</span>`;
        return `
          <article class="card">
            <div class="card-head">
              <div>
                <h3 class="card-title">${esc(n.nombre_negocio)}</h3>
                <p class="card-sub">${esc(n.rubro)}</p>
              </div>
              ${badgePausados}
            </div>
            <div class="btn-row">
              <span class="pill neutral mono">${servs} servicios</span>
              <span class="pill info mono">${citas} citas</span>
            </div>
            <div class="btn-row">
              <button type="button" class="btn btn-primary" data-entrar="${esc(n.id)}">Administrar</button>
              <button type="button" class="btn btn-danger-soft" data-eliminar-neg="${esc(n.id)}">Eliminar</button>
            </div>
          </article>
        `;
      })
      .join("");
  } catch (err) {
    mostrarToast(err.message, "error");
  }
}

async function abrirNegocio(negocioId) {
  state.negocioId = negocioId;
  $("#sim-chat-box").innerHTML = "";
  mostrarVista("negocio");
  await refrescarResumenNegocio();
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(() => {
    if (!$("#vista-negocio").classList.contains("hidden")) {
      refrescarResumenNegocio(true);
    }
  }, 15000);
}

async function refrescarResumenNegocio(silencioso = false) {
  if (!state.negocioId) return;
  try {
    const data = await api(`/api/negocios/${encodeURIComponent(state.negocioId)}/resumen`);
    state.resumen = data;
    const neg = data.negocio;
    $("#topbar-title").textContent = neg.nombre_negocio;
    $("#topbar-sub").textContent = state.rol === "superadmin" ? "Vista Super-Admin" : "Panel Dueño";
    $("#neg-nombre").textContent = neg.nombre_negocio;
    $("#neg-rubro").textContent = neg.rubro;
    $("#neg-estado-ia").textContent = data.integracion.gemini_activo
      ? `${data.integracion.motor_ia} Activo`
      : "Falta GEMINI_API_KEY";

    $("#kpi-pausados").textContent = Number(data.kpis.chats_pausados || 0);
    $("#kpi-citas").textContent = Number(data.kpis.citas_activas || 0);
    $("#kpi-servicios").textContent = Number(data.kpis.servicios_activos || 0);
    $("#kpi-promos").textContent = Number(data.kpis.promociones_activas || 0);

    renderChatsPausados(data.chats_pausados, neg.moneda);
    renderAgenda(data.agendamientos, neg.zona_horaria || "America/Bogota");
    renderServicios(data.servicios, neg.moneda);
    renderPromos(data.promociones);
    if (!silencioso) {
      llenarFormularioConfig(neg, data.integracion);
    }
  } catch (err) {
    if (!silencioso) mostrarToast(err.message, "error");
  }
}

function renderChatsPausados(chats, moneda) {
  const cont = $("#lista-chats-pausados");
  if (!chats || chats.length === 0) {
    cont.innerHTML = `<div class="empty-state"><h3>Sin chats pausados</h3><p>Tu asistente está atendiendo automáticamente. Aquí aparecerá cualquier cliente que requiera tu atención.</p></div>`;
    return;
  }
  cont.innerHTML = chats
    .map((c) => {
      const linkWa = c.enlace_whatsapp && c.enlace_whatsapp.startsWith("https://wa.me/")
        ? `<a class="btn btn-primary" href="${esc(c.enlace_whatsapp)}" target="_blank" rel="noopener noreferrer">Responder en WhatsApp</a>`
        : "";
      const hist = (c.historial || [])
        .slice(-4)
        .map((m) => `<div class="bubble ${m.rol === "cliente" ? "cliente" : "asistente"}">${esc(m.texto || m.contenido || "")}</div>`)
        .join("");
      return `
        <article class="card card-urgent">
          <div class="card-head">
            <div>
              <h3 class="card-title">${esc(c.nombre)}</h3>
              <p class="card-sub mono">${esc(c.id_cliente)} · ${esc(c.canal)}</p>
            </div>
            <span class="pill danger">Pausado</span>
          </div>
          <p class="card-sub"><strong>Resumen:</strong> ${esc(c.resumen_ia || c.motivo)}</p>
          ${c.ultima_cotizacion ? `<span class="pill brand mono">Última cotización: ${formatoMoneda(c.ultima_cotizacion, moneda)}</span>` : ""}
          <div class="chat-box" style="max-height:180px;">${hist || '<span class="card-sub">Sin historial reciente</span>'}</div>
          <div class="btn-row">
            ${linkWa}
            <button type="button" class="btn btn-success" data-reactivar="${esc(c.id_cliente)}">Reactivar Bot</button>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderAgenda(citas, zonaHoraria = "America/Bogota") {
  const cont = $("#lista-agendamientos");
  const activas = (citas || []).filter((c) => c.estado !== "cancelada");
  if (activas.length === 0) {
    cont.innerHTML = `<div class="empty-state"><h3>Agenda libre</h3><p>Aún no hay citas confirmadas para este negocio.</p></div>`;
    return;
  }
  cont.innerHTML = activas
    .map((c) => {
      const inicio = formatearFechaLocal(c.fecha_inicio, zonaHoraria);
      return `
        <article class="card">
          <div class="card-head">
            <div>
              <h3 class="card-title">${esc(c.servicio || "Cita")}</h3>
              <p class="card-sub">${esc(c.nombre_cliente)} · ${esc(c.canal)}</p>
            </div>
            <span class="pill success">${esc(c.estado)}</span>
          </div>
          <div class="btn-row">
            <span class="pill info mono">${esc(inicio)}</span>
            <span class="pill neutral">Calendar: ${esc(c.google_calendar_estado)}</span>
          </div>
          <div class="btn-row">
            <button type="button" class="btn btn-danger-soft" data-cancelar-cita="${esc(c.id)}">Cancelar cita</button>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderServicios(servicios, moneda) {
  const cont = $("#lista-servicios");
  if (!servicios || servicios.length === 0) {
    cont.innerHTML = `<div class="empty-state"><h3>Sin servicios cargados</h3><p>Agrega tus servicios y precios arriba para que el asistente pueda cotizarlos.</p></div>`;
    return;
  }
  cont.innerHTML = servicios
    .map(
      (s) => `
      <article class="card">
        <div class="card-head">
          <div>
            <h3 class="card-title">${esc(s.nombre)}</h3>
            <p class="card-sub">${esc(s.descripcion || "Sin descripción")}</p>
          </div>
          <span class="pill brand mono">${formatoMoneda(s.tarifa_unitaria, moneda)}</span>
        </div>
        <div class="btn-row">
          <span class="pill neutral mono">${Number(s.duracion_minutos || 60)} min</span>
          <span class="pill neutral mono">${esc(s.clave)}</span>
        </div>
        <div class="btn-row">
          <button type="button" class="btn" data-editar-srv="${esc(s.clave)}">Editar</button>
          <button type="button" class="btn btn-danger-soft" data-eliminar-srv="${esc(s.clave)}">Eliminar</button>
        </div>
      </article>`
    )
    .join("");
}

function renderPromos(promos) {
  const cont = $("#lista-promos");
  if (!promos || promos.length === 0) {
    cont.innerHTML = `<div class="empty-state"><h3>Sin promociones activas</h3><p>Opcionalmente crea combos o descuentos en el formulario superior.</p></div>`;
    return;
  }
  cont.innerHTML = promos
    .map(
      (p) => `
      <article class="card">
        <div class="card-head">
          <div>
            <h3 class="card-title mono">${esc(p.codigo)}</h3>
            <p class="card-sub">${esc(p.descripcion)}</p>
          </div>
          <span class="pill warning mono">Hasta ${Number(p.descuento_max ?? 0)}%</span>
        </div>
        <div class="btn-row">
          <button type="button" class="btn btn-danger-soft" data-eliminar-promo="${esc(p.codigo)}">Eliminar</button>
        </div>
      </article>`
    )
    .join("");
}

function renderDiasSelector() {
  const cont = $("#cfg-dias-row");
  cont.innerHTML = DIAS.map(
    (d) =>
      `<button type="button" class="day-pill ${state.diasSeleccionados.includes(d.id) ? "selected" : ""}" data-dia="${d.id}">${d.lbl}</button>`
  ).join("");
}

function llenarFormularioConfig(neg, integracion) {
  const esAdmin = state.rol === "superadmin";
  $("#cfg-nombre").value = neg.nombre_negocio || "";
  $("#cfg-rubro").value = neg.rubro || "";
  $("#cfg-tel-dueno").value = neg.telefono_dueno || "";
  $("#cfg-wa-negocio").value = neg.whatsapp_negocio || "";
  $("#cfg-wa-phone-id").value = neg.whatsapp_phone_number_id || "";
  $("#cfg-wa-negocio").disabled = !esAdmin;
  $("#cfg-wa-phone-id").disabled = !esAdmin;
  $("#cfg-calendar-id").value = neg.google_calendar_id || "";
  const hor = neg.horario_laboral || {};
  $("#cfg-hora-ini").value = hor.inicio || "08:00";
  $("#cfg-hora-fin").value = hor.fin || "18:00";
  state.diasSeleccionados = Array.isArray(hor.dias) ? [...hor.dias] : [1, 2, 3, 4, 5, 6];
  renderDiasSelector();

  const reg = neg.reglas_y_faq || {};
  $("#cfg-direccion").value = reg.direccion || "";
  $("#cfg-pagos").value = reg.metodos_pago || "";
  $("#cfg-politicas").value = reg.politicas || "";
  $("#cfg-tono").value = reg.tono_marca || "";
  $("#cfg-pin").value = "";
  $("#cfg-webhook-url").textContent = integracion.webhook_botpress || "";
}

function cambiarTab(tab) {
  state.tabActual = tab;
  document.querySelectorAll(".tab-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tab);
  });
  ["alertas", "agenda", "servicios", "config"].forEach((t) => {
    $(`#tab-${t}`).classList.toggle("hidden", t !== tab);
  });
}

function aplicarBorradorAFormularioNuevo(b) {
  if (!b) return;
  if (b.nombre_negocio) $("#new-nombre").value = b.nombre_negocio;
  if (b.rubro) $("#new-rubro").value = b.rubro;
  if (b.telefono_dueno) $("#new-tel-dueno").value = b.telefono_dueno;
  if (b.horario_laboral?.inicio) $("#new-hora-ini").value = b.horario_laboral.inicio;
  if (b.horario_laboral?.fin) $("#new-hora-fin").value = b.horario_laboral.fin;
  if (b.reglas_y_faq?.metodos_pago) $("#new-pagos").value = b.reglas_y_faq.metodos_pago;
  if (b.reglas_y_faq?.direccion) $("#new-direccion").value = b.reglas_y_faq.direccion;
  if (Array.isArray(b.servicios) && b.servicios.length > 0) {
    $("#new-servicios-json").value = b.servicios
      .map((s) => `${s.nombre} | ${s.tarifa_unitaria || 0} | ${s.duracion_minutos || 60} | ${s.descripcion || ""}`)
      .join("\n");
  }
}

function parsearLineasServicios(texto) {
  return String(texto || "")
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((linea) => {
      const [nombre, precio, duracion, descripcion] = linea.split("|").map((p) => (p || "").trim());
      return {
        nombre: nombre || "Servicio",
        tarifa_unitaria: Number(String(precio || "0").replace(/[^\d.]/g, "")) || 0,
        duracion_minutos: Math.max(15, Number(duracion) || 60),
        descripcion: descripcion || "",
        activo: true,
      };
    });
}

// ---------------------------------------------------------------------------
// Eventos UI
// ---------------------------------------------------------------------------
$("#rol-btn-dueno").addEventListener("click", () => {
  state.rolLogin = "dueno";
  $("#rol-btn-dueno").classList.add("active");
  $("#rol-btn-admin").classList.remove("active");
  $("#campo-negocio-login").classList.remove("hidden");
  $("#label-clave-login").textContent = "PIN de acceso del negocio";
  $("#input-clave-login").placeholder = "Ingresa tu PIN (ej. 1234)";
});

$("#rol-btn-admin").addEventListener("click", () => {
  state.rolLogin = "superadmin";
  $("#rol-btn-admin").classList.add("active");
  $("#rol-btn-dueno").classList.remove("active");
  $("#campo-negocio-login").classList.add("hidden");
  $("#label-clave-login").textContent = "Clave maestra Super-Admin";
  $("#input-clave-login").placeholder = "ADMIN_ACCESS_KEY";
});

$("#form-login").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  try {
    const body = {
      rol: state.rolLogin,
      clave: $("#input-clave-login").value,
      negocio_id: state.rolLogin === "dueno" ? $("#select-negocio-login").value : null,
    };
    const res = await api("/api/auth/login", { method: "POST", body: JSON.stringify(body) });
    state.token = res.token;
    state.rol = res.rol;
    state.negocioId = res.negocio_id || "";
    localStorage.setItem("sma_token", state.token);
    localStorage.setItem("sma_rol", state.rol);
    localStorage.setItem("sma_negocio_id", state.negocioId);
    $("#input-clave-login").value = "";
    if (state.rol === "superadmin") {
      await cargarHub();
    } else {
      await abrirNegocio(state.negocioId);
    }
  } catch (err) {
    mostrarToast(err.message, "error");
  }
});

$("#btn-cerrar-sesion").addEventListener("click", cerrarSesion);
$("#btn-volver-hub").addEventListener("click", cargarHub);
$("#btn-refrescar-hub").addEventListener("click", cargarHub);
$("#btn-refrescar-negocio").addEventListener("click", () => refrescarResumenNegocio());

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => cambiarTab(btn.dataset.tab));
});

$("#hub-negocios-grid").addEventListener("click", async (ev) => {
  const btnEntrar = ev.target.closest("[data-entrar]");
  if (btnEntrar) {
    await abrirNegocio(btnEntrar.dataset.entrar);
    return;
  }
  const btnDel = ev.target.closest("[data-eliminar-neg]");
  if (btnDel && window.confirm(`¿Eliminar el negocio '${btnDel.dataset.eliminarNeg}' y sus datos?`)) {
    try {
      await api(`/api/admin/negocios/${encodeURIComponent(btnDel.dataset.eliminarNeg)}`, { method: "DELETE" });
      mostrarToast("Negocio eliminado", "success");
      await cargarHub();
    } catch (err) {
      mostrarToast(err.message, "error");
    }
  }
});

$("#lista-chats-pausados").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("[data-reactivar]");
  if (!btn) return;
  try {
    await api(
      `/api/negocios/${encodeURIComponent(state.negocioId)}/chats/${encodeURIComponent(btn.dataset.reactivar)}/reactivar`,
      { method: "POST" }
    );
    mostrarToast("Asistente reactivado para este cliente", "success");
    await refrescarResumenNegocio();
  } catch (err) {
    mostrarToast(err.message, "error");
  }
});

$("#lista-agendamientos").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("[data-cancelar-cita]");
  if (!btn) return;
  try {
    await api(
      `/api/negocios/${encodeURIComponent(state.negocioId)}/agendamientos/${encodeURIComponent(btn.dataset.cancelarCita)}`,
      { method: "DELETE" }
    );
    mostrarToast("Cita cancelada", "success");
    await refrescarResumenNegocio();
  } catch (err) {
    mostrarToast(err.message, "error");
  }
});

$("#form-cita-manual").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  try {
    await api(`/api/negocios/${encodeURIComponent(state.negocioId)}/agendamientos`, {
      method: "POST",
      body: JSON.stringify({
        fecha_hora: $("#cita-fecha-hora").value,
        servicio: $("#cita-servicio").value,
        nombre_cliente: $("#cita-cliente").value,
        duracion_minutos: Number($("#cita-duracion").value || 60),
      }),
    });
    ev.target.reset();
    mostrarToast("Reserva guardada en la agenda", "success");
    await refrescarResumenNegocio();
  } catch (err) {
    mostrarToast(err.message, "error");
  }
});

$("#form-servicio").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  try {
    await api(`/api/negocios/${encodeURIComponent(state.negocioId)}/servicios`, {
      method: "POST",
      body: JSON.stringify({
        clave: $("#srv-clave").value || null,
        nombre: $("#srv-nombre").value,
        tarifa_unitaria: Number($("#srv-precio").value || 0),
        duracion_minutos: Number($("#srv-duracion").value || 60),
        descripcion: $("#srv-desc").value,
        activo: true,
      }),
    });
    ev.target.reset();
    $("#srv-duracion").value = 60;
    mostrarToast("Servicio guardado", "success");
    await refrescarResumenNegocio();
  } catch (err) {
    mostrarToast(err.message, "error");
  }
});

$("#btn-limpiar-srv").addEventListener("click", () => {
  $("#form-servicio").reset();
  $("#srv-duracion").value = 60;
});

$("#lista-servicios").addEventListener("click", async (ev) => {
  const btnEdit = ev.target.closest("[data-editar-srv]");
  if (btnEdit && state.resumen) {
    const srv = (state.resumen.servicios || []).find((s) => s.clave === btnEdit.dataset.editarSrv);
    if (srv) {
      $("#srv-clave").value = srv.clave;
      $("#srv-nombre").value = srv.nombre;
      $("#srv-precio").value = srv.tarifa_unitaria;
      $("#srv-duracion").value = srv.duracion_minutos || 60;
      $("#srv-desc").value = srv.descripcion || "";
      $("#srv-nombre").focus();
    }
    return;
  }
  const btnDel = ev.target.closest("[data-eliminar-srv]");
  if (btnDel) {
    try {
      await api(
        `/api/negocios/${encodeURIComponent(state.negocioId)}/servicios/${encodeURIComponent(btnDel.dataset.eliminarSrv)}`,
        { method: "DELETE" }
      );
      mostrarToast("Servicio eliminado", "success");
      await refrescarResumenNegocio();
    } catch (err) {
      mostrarToast(err.message, "error");
    }
  }
});

$("#form-promo").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  try {
    await api(`/api/negocios/${encodeURIComponent(state.negocioId)}/promociones`, {
      method: "POST",
      body: JSON.stringify({
        codigo: $("#promo-codigo").value,
        descripcion: $("#promo-desc").value,
        descuento_max: Number($("#promo-pct").value ?? 10),
        activa: true,
        condiciones: { cantidad_minima: Number($("#promo-min").value ?? 2) },
      }),
    });
    ev.target.reset();
    mostrarToast("Promoción guardada", "success");
    await refrescarResumenNegocio();
  } catch (err) {
    mostrarToast(err.message, "error");
  }
});

$("#lista-promos").addEventListener("click", async (ev) => {
  const btnDel = ev.target.closest("[data-eliminar-promo]");
  if (!btnDel) return;
  try {
    await api(
      `/api/negocios/${encodeURIComponent(state.negocioId)}/promociones/${encodeURIComponent(btnDel.dataset.eliminarPromo)}`,
      { method: "DELETE" }
    );
    mostrarToast("Promoción eliminada", "success");
    await refrescarResumenNegocio();
  } catch (err) {
    mostrarToast(err.message, "error");
  }
});

$("#cfg-dias-row").addEventListener("click", (ev) => {
  const btn = ev.target.closest("[data-dia]");
  if (!btn) return;
  const dia = Number(btn.dataset.dia);
  if (state.diasSeleccionados.includes(dia)) {
    state.diasSeleccionados = state.diasSeleccionados.filter((d) => d !== dia);
  } else {
    state.diasSeleccionados = [...state.diasSeleccionados, dia].sort((a, b) => a - b);
  }
  renderDiasSelector();
});

$("#form-config-negocio").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const negActual = state.resumen?.negocio || {};
  const reglasActuales = negActual.reglas_y_faq || {};
  const payload = {
    nombre_negocio: $("#cfg-nombre").value,
    rubro: $("#cfg-rubro").value,
    telefono_dueno: $("#cfg-tel-dueno").value,
    google_calendar_id: $("#cfg-calendar-id").value,
    horario_laboral: {
      ...(negActual.horario_laboral || {}),
      dias: state.diasSeleccionados.length ? state.diasSeleccionados : [1, 2, 3, 4, 5, 6],
      inicio: $("#cfg-hora-ini").value || "08:00",
      fin: $("#cfg-hora-fin").value || "18:00",
      duracion_min: negActual.horario_laboral?.duracion_min || 60,
    },
    reglas_y_faq: {
      ...reglasActuales,
      direccion: $("#cfg-direccion").value,
      metodos_pago: $("#cfg-pagos").value,
      politicas: $("#cfg-politicas").value,
      tono_marca: $("#cfg-tono").value,
    },
  };
  if (state.rol === "superadmin") {
    payload.whatsapp_negocio = $("#cfg-wa-negocio").value;
    payload.whatsapp_phone_number_id = $("#cfg-wa-phone-id").value;
  }
  if ($("#cfg-pin").value.trim()) {
    payload.pin_dueno = $("#cfg-pin").value.trim();
  }
  try {
    const res = await api(`/api/negocios/${encodeURIComponent(state.negocioId)}/configuracion`, {
      method: "PUT",
      body: JSON.stringify(payload),
    });
    if (res.nuevo_token) {
      state.token = res.nuevo_token;
      localStorage.setItem("sma_token", state.token);
    }
    mostrarToast("Configuración guardada y activa al instante", "success");
    await refrescarResumenNegocio();
  } catch (err) {
    mostrarToast(err.message, "error");
  }
});

// Simulador de chat en vivo
$("#form-sim-chat").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const input = $("#sim-input");
  const texto = input.value.trim();
  if (!texto) return;
  input.value = "";
  const box = $("#sim-chat-box");
  box.insertAdjacentHTML("beforeend", `<div class="bubble cliente">${esc(texto)}</div>`);
  box.scrollTop = box.scrollHeight;
  try {
    const res = await api(`/api/negocios/${encodeURIComponent(state.negocioId)}/probar-chat`, {
      method: "POST",
      body: JSON.stringify({ texto, id_cliente: "simulador_pwa", nombre: "Cliente Prueba" }),
    });
    const respTxt = res.respuesta || (res.pausado_humano ? "(Bot en silencio: chat pausado para atención humana)" : "");
    box.insertAdjacentHTML("beforeend", `<div class="bubble asistente">${esc(respTxt)}</div>`);
    box.scrollTop = box.scrollHeight;
    if (res.pausado_humano || res.cita_confirmada) {
      await refrescarResumenNegocio(true);
    }
  } catch (err) {
    mostrarToast(err.message, "error");
  }
});

// Modal Nuevo Negocio + Onboarding Gemini
$("#btn-nuevo-negocio").addEventListener("click", () => {
  state.onbHistorial = [];
  state.onbBorrador = null;
  $("#form-crear-negocio").reset();
  $("#new-pin").value = "1234";
  $("#onb-chat-box").innerHTML = `<div class="bubble asistente">${esc(SALUDO_ONBOARDING)}</div>`;
  $("#modal-onboarding").classList.remove("hidden");
});

$("#btn-cerrar-modal").addEventListener("click", () => {
  $("#modal-onboarding").classList.add("hidden");
});

$("#form-onb-chat").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const input = $("#onb-input");
  const msg = input.value.trim();
  if (!msg) return;
  input.value = "";
  const box = $("#onb-chat-box");
  box.insertAdjacentHTML("beforeend", `<div class="bubble cliente">${esc(msg)}</div>`);
  box.scrollTop = box.scrollHeight;
  try {
    const res = await api("/api/onboarding/chat", {
      method: "POST",
      body: JSON.stringify({
        historial: state.onbHistorial,
        mensaje: msg,
        borrador_actual: state.onbBorrador,
      }),
    });
    state.onbHistorial.push({ rol: "usuario", texto: msg });
    state.onbHistorial.push({ rol: "asistente", texto: res.respuesta_chat });
    state.onbBorrador = res.borrador_negocio;
    aplicarBorradorAFormularioNuevo(res.borrador_negocio);
    box.insertAdjacentHTML("beforeend", `<div class="bubble asistente">${esc(res.respuesta_chat)}</div>`);
    box.scrollTop = box.scrollHeight;
  } catch (err) {
    mostrarToast(err.message, "error");
  }
});

$("#form-crear-negocio").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  try {
    const servicios = parsearLineasServicios($("#new-servicios-json").value);
    const b = state.onbBorrador || {};
    await api("/api/admin/negocios", {
      method: "POST",
      body: JSON.stringify({
        id: b.id || null,
        nombre_negocio: $("#new-nombre").value,
        rubro: $("#new-rubro").value,
        moneda: b.moneda || "COP",
        unidad_cantidad: b.unidad_cantidad || "servicios",
        telefono_dueno: $("#new-tel-dueno").value,
        whatsapp_negocio: b.whatsapp_negocio || "",
        google_calendar_id: b.google_calendar_id || "",
        pin_dueno: $("#new-pin").value || "1234",
        horario_laboral: {
          dias: b.horario_laboral?.dias || [1, 2, 3, 4, 5, 6],
          inicio: $("#new-hora-ini").value || "09:00",
          fin: $("#new-hora-fin").value || "19:00",
          duracion_min: b.horario_laboral?.duracion_min || 60,
        },
        reglas_y_faq: {
          ...(b.reglas_y_faq || {}),
          metodos_pago: $("#new-pagos").value,
          direccion: $("#new-direccion").value,
        },
        servicios,
        promociones: (b.promociones || []).map((p) => ({
          codigo: p.codigo,
          descripcion: p.descripcion,
          descuento_max: p.descuento_max ?? 10,
          activa: true,
          condiciones: { cantidad_minima: p.cantidad_minima ?? 2 },
        })),
      }),
    });
    $("#modal-onboarding").classList.add("hidden");
    mostrarToast("Negocio creado y activado correctamente", "success");
    await cargarHub();
  } catch (err) {
    mostrarToast(err.message, "error");
  }
});

// Arranque inicial + registro PWA
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}

(async function init() {
  await cargarNegociosPublicos();
  if (state.token && state.rol === "superadmin") {
    await cargarHub();
  } else if (state.token && state.rol === "dueno" && state.negocioId) {
    await abrirNegocio(state.negocioId);
  } else {
    mostrarVista("login");
  }
})();
