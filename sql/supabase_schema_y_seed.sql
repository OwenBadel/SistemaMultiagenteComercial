-- =============================================================================
-- SistemaMultiagenteComercial - Esquema Relacional y Datos Iniciales Supabase
-- Instrucciones:
--   1. Abre tu proyecto en Supabase -> SQL Editor -> New Query.
--   2. Pega y ejecuta este script completo una sola vez.
--   3. Configura SUPABASE_URL, SUPABASE_KEY (service_role) y NEGOCIO_ID en .env.
-- =============================================================================

CREATE TABLE IF NOT EXISTS negocios (
    id TEXT PRIMARY KEY,
    nombre_negocio TEXT NOT NULL,
    rubro TEXT NOT NULL,
    moneda TEXT NOT NULL DEFAULT 'COP',
    unidad_cantidad TEXT NOT NULL DEFAULT 'horas',
    etiqueta_cantidad TEXT NOT NULL DEFAULT 'horas_estimadas',
    etiqueta_complejidad TEXT NOT NULL DEFAULT 'complejidad',
    telefono_dueno TEXT DEFAULT '',
    margen_base DOUBLE PRECISION NOT NULL DEFAULT 0.35,
    margen_minimo DOUBLE PRECISION NOT NULL DEFAULT 0.22,
    descuento_maximo DOUBLE PRECISION NOT NULL DEFAULT 15.0,
    mensaje_catalogo_intro TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS servicios (
    negocio_id TEXT NOT NULL REFERENCES negocios(id) ON DELETE CASCADE,
    clave TEXT NOT NULL,
    servicio_id_num INTEGER NOT NULL,
    nombre TEXT NOT NULL,
    descripcion TEXT NOT NULL,
    tarifa_unitaria DOUBLE PRECISION NOT NULL,
    palabras_clave JSONB NOT NULL DEFAULT '[]'::jsonb,
    requiere_complejidad BOOLEAN NOT NULL DEFAULT TRUE,
    costo_adicional_mantenimiento DOUBLE PRECISION NOT NULL DEFAULT 380000.0,
    PRIMARY KEY (negocio_id, clave)
);

CREATE TABLE IF NOT EXISTS promociones_vigentes (
    negocio_id TEXT NOT NULL REFERENCES negocios(id) ON DELETE CASCADE,
    codigo TEXT NOT NULL,
    descripcion TEXT NOT NULL,
    descuento_max DOUBLE PRECISION NOT NULL,
    activa BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (negocio_id, codigo)
);

CREATE TABLE IF NOT EXISTS tarifas_historicas (
    negocio_id TEXT NOT NULL REFERENCES negocios(id) ON DELETE CASCADE,
    id_cotizacion TEXT NOT NULL,
    tipo_servicio TEXT NOT NULL,
    tipo_servicio_id INTEGER NOT NULL,
    complejidad INTEGER NOT NULL,
    cantidad_unidades INTEGER NOT NULL,
    urgencia_dias INTEGER NOT NULL,
    incluye_adicional INTEGER NOT NULL DEFAULT 0,
    costo_base DOUBLE PRECISION NOT NULL,
    costo_total DOUBLE PRECISION NOT NULL,
    margen_estimado DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (negocio_id, id_cotizacion)
);

CREATE TABLE IF NOT EXISTS historico_ofertas (
    negocio_id TEXT NOT NULL REFERENCES negocios(id) ON DELETE CASCADE,
    id_oferta TEXT NOT NULL,
    monto_cotizado DOUBLE PRECISION NOT NULL,
    nivel_indecision DOUBLE PRECISION NOT NULL,
    complejidad INTEGER NOT NULL,
    fidelidad_cliente INTEGER NOT NULL,
    porcentaje_descuento_optimo DOUBLE PRECISION NOT NULL,
    cerrado_exitosamente INTEGER NOT NULL,
    PRIMARY KEY (negocio_id, id_oferta)
);

CREATE TABLE IF NOT EXISTS perfiles_clientes (
    negocio_id TEXT NOT NULL REFERENCES negocios(id) ON DELETE CASCADE,
    id_cliente TEXT NOT NULL,
    nombre TEXT NOT NULL,
    canal TEXT NOT NULL,
    historial_interacciones JSONB NOT NULL DEFAULT '[]'::jsonb,
    tono_actual TEXT NOT NULL DEFAULT 'neutral',
    fidelidad INTEGER NOT NULL DEFAULT 1,
    ultima_cotizacion DOUBLE PRECISION,
    ultima_complejidad INTEGER NOT NULL DEFAULT 3,
    parametros_parciales JSONB NOT NULL DEFAULT '{}'::jsonb,
    esperando_parametros_cotizacion BOOLEAN NOT NULL DEFAULT FALSE,
    promocion_aplicada DOUBLE PRECISION,
    automatizacion_activa BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (negocio_id, id_cliente)
);

CREATE TABLE IF NOT EXISTS alertas_humanas (
    id BIGSERIAL PRIMARY KEY,
    negocio_id TEXT NOT NULL REFERENCES negocios(id) ON DELETE CASCADE,
    timestamp TEXT NOT NULL,
    id_cliente TEXT NOT NULL,
    nombre TEXT NOT NULL,
    canal TEXT NOT NULL,
    motivo_critico TEXT NOT NULL,
    historial_charla JSONB NOT NULL DEFAULT '[]'::jsonb,
    estado TEXT NOT NULL,
    resuelta BOOLEAN NOT NULL DEFAULT FALSE
);

-- =============================================================================
-- NEGOCIO 1: tech_freelance (Servicios de Desarrollo e IA - cotiza por horas)
-- =============================================================================
INSERT INTO negocios (
    id, nombre_negocio, rubro, moneda, unidad_cantidad, etiqueta_cantidad,
    etiqueta_complejidad, telefono_dueno, margen_base, margen_minimo,
    descuento_maximo, mensaje_catalogo_intro
) VALUES (
    'tech_freelance',
    'Consultoria Digital Independiente',
    'Desarrollo de Software y Automatizacion',
    'COP',
    'horas',
    'horas_estimadas',
    'complejidad',
    '573001234567',
    0.35,
    0.22,
    15.0,
    'Con gusto te comparto nuestro portafolio de servicios digitales'
) ON CONFLICT (id) DO NOTHING;

INSERT INTO servicios (
    negocio_id, clave, servicio_id_num, nombre, descripcion, tarifa_unitaria,
    palabras_clave, requiere_complejidad, costo_adicional_mantenimiento
) VALUES
    ('tech_freelance', 'desarrollo_web', 1, 'Desarrollo Web Corporativo / Landing & Portal', 'Sitio web responsivo, panel administrativo y despliegue en nube.', 85000, '["web","pagina","pÃ¡gina","sitio","landing"]'::jsonb, TRUE, 380000),
    ('tech_freelance', 'automatizacion_ia', 2, 'Automatizacion Inteligente con Flujos IA & Bots', 'Integracion de asistentes conversacionales, extraccion de datos y APIs.', 125000, '["automatiz","bot","ia","inteligencia artificial"]'::jsonb, TRUE, 380000),
    ('tech_freelance', 'ecommerce', 3, 'Tienda Virtual E-Commerce con Pasarela de Pagos', 'Catalogo interactivo, carrito, inventario y pagos en linea.', 95000, '["tienda","carrito","ecommerce","e-commerce"]'::jsonb, TRUE, 380000),
    ('tech_freelance', 'app_movil', 4, 'Aplicacion Movil Multiplataforma (Android / iOS)', 'App hibrida conectada a backend REST con notificaciones push.', 115000, '["movil","mÃ³vil","app","android","ios"]'::jsonb, TRUE, 380000),
    ('tech_freelance', 'consultoria_cloud', 5, 'Arquitectura Cloud, Migracion y DevOps', 'Contenedores Docker, CI/CD, seguridad y monitoreo en nube.', 110000, '["cloud","nube","devops","servidor"]'::jsonb, TRUE, 380000)
ON CONFLICT (negocio_id, clave) DO NOTHING;

INSERT INTO promociones_vigentes (negocio_id, codigo, descripcion, descuento_max, activa) VALUES
    ('tech_freelance', 'PROMO_CIERRE_10', 'Incentivo de cierre rapido para clientes indecisos', 12.0, TRUE),
    ('tech_freelance', 'PROMO_FIDELIDAD_15', 'Beneficio tope de retencion comercial', 15.0, TRUE)
ON CONFLICT (negocio_id, codigo) DO NOTHING;

INSERT INTO tarifas_historicas (
    negocio_id, id_cotizacion, tipo_servicio, tipo_servicio_id, complejidad,
    cantidad_unidades, urgencia_dias, incluye_adicional, costo_base, costo_total, margen_estimado
)
SELECT
    'tech_freelance',
    'COT-' || LPAD(g::text, 3, '0'),
    CASE ((g - 1) % 5)
        WHEN 0 THEN 'desarrollo_web'
        WHEN 1 THEN 'automatizacion_ia'
        WHEN 2 THEN 'ecommerce'
        WHEN 3 THEN 'app_movil'
        ELSE 'consultoria_cloud'
    END,
    ((g - 1) % 5) + 1,
    ((g * 3) % 5) + 1,
    15 + ((g * 7) % 120),
    5 + ((g * 4) % 35),
    (g % 2),
    ROUND(((15 + ((g * 7) % 120)) * (CASE ((g - 1) % 5) WHEN 0 THEN 85000 WHEN 1 THEN 125000 WHEN 2 THEN 95000 WHEN 3 THEN 115000 ELSE 110000 END) * (1.0 + 0.14 * (((g * 3) % 5))))::numeric, 2),
    ROUND((((15 + ((g * 7) % 120)) * (CASE ((g - 1) % 5) WHEN 0 THEN 85000 WHEN 1 THEN 125000 WHEN 2 THEN 95000 WHEN 3 THEN 115000 ELSE 110000 END) * (1.0 + 0.14 * (((g * 3) % 5)))) + (g % 2) * 380000)::numeric, -3),
    0.35
FROM generate_series(1, 45) AS g
ON CONFLICT (negocio_id, id_cotizacion) DO NOTHING;

INSERT INTO historico_ofertas (
    negocio_id, id_oferta, monto_cotizado, nivel_indecision, complejidad,
    fidelidad_cliente, porcentaje_descuento_optimo, cerrado_exitosamente
)
SELECT
    'tech_freelance',
    'OFE-' || LPAD(g::text, 3, '0'),
    2000000 + (g * 250000),
    CASE
        WHEN g <= 12 THEN ROUND((0.18 + (g % 5) * 0.05)::numeric, 2)
        WHEN g <= 34 THEN ROUND((0.60 + (g % 6) * 0.04)::numeric, 2)
        ELSE ROUND((0.89 + (g % 4) * 0.02)::numeric, 2)
    END,
    ((g * 2) % 5) + 1,
    CASE WHEN g <= 34 THEN ((g % 2) + 1) ELSE 3 END,
    CASE
        WHEN g <= 12 THEN ROUND((4.0 + (g % 5) * 0.5)::numeric, 2)
        WHEN g <= 34 THEN ROUND((8.5 + (g % 6) * 0.6)::numeric, 2)
        ELSE ROUND((18.0 + (g % 4) * 1.1)::numeric, 2)
    END,
    1
FROM generate_series(1, 45) AS g
ON CONFLICT (negocio_id, id_oferta) DO NOTHING;

-- =============================================================================
-- NEGOCIO 2 (Ejemplo de otro independiente): estudio_fotografia (cotiza por sesiones)
-- Para probarlo, cambia NEGOCIO_ID=estudio_fotografia en tu archivo .env
-- =============================================================================
INSERT INTO negocios (
    id, nombre_negocio, rubro, moneda, unidad_cantidad, etiqueta_cantidad,
    etiqueta_complejidad, telefono_dueno, margen_base, margen_minimo,
    descuento_maximo, mensaje_catalogo_intro
) VALUES (
    'estudio_fotografia',
    'Estudio Visual Independiente',
    'Fotografia Comercial y Produccion de Video',
    'COP',
    'sesiones',
    'sesiones_requeridas',
    'complejidad',
    '573001234567',
    0.40,
    0.25,
    12.0,
    'Hola! Aqui tienes nuestros paquetes de fotografia y produccion visual'
) ON CONFLICT (id) DO NOTHING;

INSERT INTO servicios (
    negocio_id, clave, servicio_id_num, nombre, descripcion, tarifa_unitaria,
    palabras_clave, requiere_complejidad, costo_adicional_mantenimiento
) VALUES
    ('estudio_fotografia', 'sesion_producto', 1, 'Fotografia de Producto E-Commerce', 'Iluminacion en estudio, fondo limpio y retoque editorial.', 280000, '["producto","catalogo","catÃ¡logo","articulo","ecommerce"]'::jsonb, TRUE, 150000),
    ('estudio_fotografia', 'retrato_corporativo', 2, 'Retrato Corporativo y Marca Personal', 'Sesion profesional en estudio o locacion con entrega digital.', 320000, '["retrato","corporativo","perfil","marca personal","foto"]'::jsonb, FALSE, 120000),
    ('estudio_fotografia', 'cobertura_evento', 3, 'Cobertura Fotografica de Eventos', 'Registro completo en locacion con edicion de galeria.', 450000, '["evento","boda","grado","conferencia","fiesta"]'::jsonb, TRUE, 200000),
    ('estudio_fotografia', 'video_reel', 4, 'Produccion de Video Reels / Comercial Corto', 'Grabacion 4K, edicion dinamica y colorizacion para redes.', 390000, '["video","reel","reels","comercial","grabacion"]'::jsonb, TRUE, 180000)
ON CONFLICT (negocio_id, clave) DO NOTHING;

INSERT INTO promociones_vigentes (negocio_id, codigo, descripcion, descuento_max, activa) VALUES
    ('estudio_fotografia', 'PACK_SESIONES_10', 'Descuento para paquetes multi-sesion', 10.0, TRUE)
ON CONFLICT (negocio_id, codigo) DO NOTHING;

INSERT INTO tarifas_historicas (
    negocio_id, id_cotizacion, tipo_servicio, tipo_servicio_id, complejidad,
    cantidad_unidades, urgencia_dias, incluye_adicional, costo_base, costo_total, margen_estimado
)
SELECT
    'estudio_fotografia',
    'FOT-' || LPAD(g::text, 3, '0'),
    CASE ((g - 1) % 4)
        WHEN 0 THEN 'sesion_producto'
        WHEN 1 THEN 'retrato_corporativo'
        WHEN 2 THEN 'cobertura_evento'
        ELSE 'video_reel'
    END,
    ((g - 1) % 4) + 1,
    ((g * 3) % 5) + 1,
    1 + (g % 8),
    3 + ((g * 3) % 20),
    (g % 2),
    ROUND(((1 + (g % 8)) * (CASE ((g - 1) % 4) WHEN 0 THEN 280000 WHEN 1 THEN 320000 WHEN 2 THEN 450000 ELSE 390000 END) * (1.0 + 0.12 * (((g * 3) % 5))))::numeric, 2),
    ROUND((((1 + (g % 8)) * (CASE ((g - 1) % 4) WHEN 0 THEN 280000 WHEN 1 THEN 320000 WHEN 2 THEN 450000 ELSE 390000 END) * (1.0 + 0.12 * (((g * 3) % 5)))) + (g % 2) * 150000)::numeric, -3),
    0.40
FROM generate_series(1, 35) AS g
ON CONFLICT (negocio_id, id_cotizacion) DO NOTHING;

INSERT INTO historico_ofertas (
    negocio_id, id_oferta, monto_cotizado, nivel_indecision, complejidad,
    fidelidad_cliente, porcentaje_descuento_optimo, cerrado_exitosamente
)
SELECT
    'estudio_fotografia',
    'OFE-FOT-' || LPAD(g::text, 3, '0'),
    400000 + (g * 90000),
    CASE
        WHEN g <= 10 THEN ROUND((0.20 + (g % 4) * 0.05)::numeric, 2)
        WHEN g <= 26 THEN ROUND((0.62 + (g % 5) * 0.04)::numeric, 2)
        ELSE ROUND((0.90 + (g % 3) * 0.02)::numeric, 2)
    END,
    ((g * 2) % 5) + 1,
    CASE WHEN g <= 26 THEN ((g % 2) + 1) ELSE 3 END,
    CASE
        WHEN g <= 10 THEN ROUND((3.5 + (g % 4) * 0.5)::numeric, 2)
        WHEN g <= 26 THEN ROUND((7.0 + (g % 5) * 0.6)::numeric, 2)
        ELSE ROUND((14.0 + (g % 3) * 1.2)::numeric, 2)
    END,
    1
FROM generate_series(1, 35) AS g
ON CONFLICT (negocio_id, id_oferta) DO NOTHING;
