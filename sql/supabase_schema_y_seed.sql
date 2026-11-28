-- =============================================================================
-- SistemaMultiagenteComercial - Esquema Relacional y Semillas Oficiales (Supabase)
-- Instrucciones:
--   1. Abre Supabase -> SQL Editor -> New Query.
--   2. Pega y ejecuta este script completo una sola vez.
--   3. Configura SUPABASE_URL y SUPABASE_KEY (secret / service_role) en .env.
-- ADVERTENCIA: Elimina y recrea todas las tablas desde cero (ideal para instalar
-- o limpiar una base de datos anterior). No ejecutar si ya tienes citas reales.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS btree_gist;

-- Limpieza total de tablas anteriores (incluyendo tablas legacy de ML)
DROP TABLE IF EXISTS public.agendamientos CASCADE;
DROP TABLE IF EXISTS public.alertas_humanas CASCADE;
DROP TABLE IF EXISTS public.perfiles_clientes CASCADE;
DROP TABLE IF EXISTS public.promociones_vigentes CASCADE;
DROP TABLE IF EXISTS public.servicios CASCADE;
DROP TABLE IF EXISTS public.tarifas_historicas CASCADE;
DROP TABLE IF EXISTS public.historico_ofertas CASCADE;
DROP TABLE IF EXISTS public.negocios CASCADE;

-- 1. Negocios (multi-tenant)
CREATE TABLE IF NOT EXISTS public.negocios (
    id TEXT PRIMARY KEY,
    nombre_negocio TEXT NOT NULL,
    rubro TEXT NOT NULL,
    moneda TEXT NOT NULL DEFAULT 'COP',
    unidad_cantidad TEXT NOT NULL DEFAULT 'servicios',
    etiqueta_cantidad TEXT NOT NULL DEFAULT 'cantidad',
    etiqueta_complejidad TEXT NOT NULL DEFAULT 'complejidad',
    telefono_dueno TEXT NOT NULL DEFAULT '',
    whatsapp_negocio TEXT NOT NULL DEFAULT '',
    whatsapp_phone_number_id TEXT NOT NULL DEFAULT '',
    whatsapp_access_token TEXT NOT NULL DEFAULT '',
    botpress_bot_id TEXT NOT NULL DEFAULT '',
    google_calendar_id TEXT NOT NULL DEFAULT '',
    zona_horaria TEXT NOT NULL DEFAULT 'America/Bogota',
    margen_base DOUBLE PRECISION NOT NULL DEFAULT 0.35,
    margen_minimo DOUBLE PRECISION NOT NULL DEFAULT 0.22,
    descuento_maximo DOUBLE PRECISION NOT NULL DEFAULT 15.0,
    mensaje_catalogo_intro TEXT NOT NULL DEFAULT 'Con gusto te comparto nuestros servicios',
    activo BOOLEAN NOT NULL DEFAULT TRUE,
    horario_laboral JSONB NOT NULL DEFAULT '{"dias":[1,2,3,4,5,6],"inicio":"08:00","fin":"18:00","duracion_min":60}'::jsonb,
    reglas_y_faq JSONB NOT NULL DEFAULT '{}'::jsonb,
    pin_dueno_hash TEXT NOT NULL DEFAULT ''
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_negocios_wa_phone_id
    ON public.negocios (whatsapp_phone_number_id)
    WHERE whatsapp_phone_number_id <> '';

CREATE UNIQUE INDEX IF NOT EXISTS idx_negocios_wa_numero
    ON public.negocios (whatsapp_negocio)
    WHERE whatsapp_negocio <> '';

CREATE UNIQUE INDEX IF NOT EXISTS idx_negocios_botpress_id
    ON public.negocios (botpress_bot_id)
    WHERE botpress_bot_id <> '';

-- 2. Catalogo de servicios por negocio
CREATE TABLE IF NOT EXISTS public.servicios (
    negocio_id TEXT NOT NULL REFERENCES public.negocios(id) ON DELETE CASCADE,
    clave TEXT NOT NULL,
    servicio_id_num INTEGER NOT NULL DEFAULT 1,
    nombre TEXT NOT NULL,
    descripcion TEXT NOT NULL DEFAULT '',
    tarifa_unitaria DOUBLE PRECISION NOT NULL,
    palabras_clave JSONB NOT NULL DEFAULT '[]'::jsonb,
    requiere_complejidad BOOLEAN NOT NULL DEFAULT FALSE,
    costo_adicional_mantenimiento DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    duracion_minutos INTEGER NOT NULL DEFAULT 60,
    activo BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (negocio_id, clave)
);

-- 3. Promociones vigentes por negocio
CREATE TABLE IF NOT EXISTS public.promociones_vigentes (
    negocio_id TEXT NOT NULL REFERENCES public.negocios(id) ON DELETE CASCADE,
    codigo TEXT NOT NULL,
    descripcion TEXT NOT NULL,
    descuento_max DOUBLE PRECISION NOT NULL,
    activa BOOLEAN NOT NULL DEFAULT TRUE,
    condiciones JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (negocio_id, codigo)
);

-- 4. Perfiles de clientes y estado de automatizacion por chat
CREATE TABLE IF NOT EXISTS public.perfiles_clientes (
    negocio_id TEXT NOT NULL REFERENCES public.negocios(id) ON DELETE CASCADE,
    id_cliente TEXT NOT NULL,
    nombre TEXT NOT NULL,
    canal TEXT NOT NULL,
    historial_interacciones JSONB NOT NULL DEFAULT '[]'::jsonb,
    tono_actual TEXT NOT NULL DEFAULT 'neutral',
    fidelidad INTEGER NOT NULL DEFAULT 1,
    ultima_cotizacion DOUBLE PRECISION,
    parametros_parciales JSONB NOT NULL DEFAULT '{}'::jsonb,
    automatizacion_activa BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (negocio_id, id_cliente)
);

-- 5. Alertas de escalamiento Human-in-the-Loop
CREATE TABLE IF NOT EXISTS public.alertas_humanas (
    id BIGSERIAL PRIMARY KEY,
    negocio_id TEXT NOT NULL REFERENCES public.negocios(id) ON DELETE CASCADE,
    timestamp TEXT NOT NULL,
    id_cliente TEXT NOT NULL,
    nombre TEXT NOT NULL,
    canal TEXT NOT NULL,
    motivo_critico TEXT NOT NULL,
    resumen_ia TEXT NOT NULL DEFAULT '',
    historial_charla JSONB NOT NULL DEFAULT '[]'::jsonb,
    estado TEXT NOT NULL DEFAULT 'pausado',
    resuelta BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_alertas_negocio_abiertas
    ON public.alertas_humanas (negocio_id, resuelta, id DESC);

-- 6. Agendamientos y sincronizacion con Google Calendar
CREATE TABLE IF NOT EXISTS public.agendamientos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    negocio_id TEXT NOT NULL REFERENCES public.negocios(id) ON DELETE CASCADE,
    id_cliente TEXT NOT NULL,
    nombre_cliente TEXT NOT NULL,
    canal TEXT NOT NULL,
    servicio TEXT,
    fecha_inicio TIMESTAMPTZ NOT NULL,
    fecha_fin TIMESTAMPTZ NOT NULL,
    estado TEXT NOT NULL DEFAULT 'confirmada'
        CHECK (estado IN ('pendiente', 'confirmada', 'cancelada', 'error')),
    registrado_en TEXT NOT NULL
        CHECK (registrado_en IN ('supabase_y_google_calendar', 'solo_supabase')),
    google_calendar_estado TEXT NOT NULL
        CHECK (google_calendar_estado IN ('pendiente', 'sincronizado', 'no_configurado', 'error')),
    google_event_id TEXT,
    google_event_link TEXT,
    google_calendar_error TEXT,
    observaciones TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT agendamientos_fecha_valida CHECK (fecha_fin > fecha_inicio)
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'agendamientos_sin_cruce_activo'
          AND conrelid = 'public.agendamientos'::regclass
    ) THEN
        ALTER TABLE public.agendamientos
            ADD CONSTRAINT agendamientos_sin_cruce_activo
            EXCLUDE USING gist (
                negocio_id WITH =,
                tstzrange(fecha_inicio, fecha_fin, '[)') WITH &&
            )
            WHERE (estado IN ('pendiente', 'confirmada'));
    END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_agendamientos_negocio ON public.agendamientos (negocio_id);
CREATE INDEX IF NOT EXISTS idx_agendamientos_cliente ON public.agendamientos (id_cliente);
CREATE INDEX IF NOT EXISTS idx_agendamientos_fecha ON public.agendamientos (negocio_id, fecha_inicio);
CREATE UNIQUE INDEX IF NOT EXISTS idx_agendamientos_google_event
    ON public.agendamientos (google_event_id)
    WHERE google_event_id IS NOT NULL;

-- 7. Seguridad multi-tenant: Row Level Security (RLS) en todas las tablas
-- Sin politicas abiertas para anon/authenticated; el backend accede con service_role / secret key.
ALTER TABLE public.negocios ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.servicios ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.promociones_vigentes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.perfiles_clientes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alertas_humanas ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agendamientos ENABLE ROW LEVEL SECURITY;

GRANT USAGE ON SCHEMA public TO service_role;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO service_role;

-- =============================================================================
-- SEMILLA 1: estudio_fotografia (Estudio Visual Independiente)
-- PIN inicial de demo = 1234 (cambiar desde el panel PWA)
-- =============================================================================
INSERT INTO public.negocios (
    id, nombre_negocio, rubro, moneda, unidad_cantidad, etiqueta_cantidad,
    etiqueta_complejidad, telefono_dueno, margen_base, margen_minimo,
    descuento_maximo, mensaje_catalogo_intro, activo, zona_horaria,
    horario_laboral, reglas_y_faq, pin_dueno_hash
) VALUES (
    'estudio_fotografia',
    'Estudio Visual Independiente',
    'Fotografia Comercial y Produccion de Video',
    'COP',
    'sesiones',
    'sesiones_requeridas',
    'complejidad',
    '',
    0.40,
    0.25,
    12.0,
    'Hola! Aqui tienes nuestros paquetes de fotografia y produccion visual',
    TRUE,
    'America/Bogota',
    '{"dias":[1,2,3,4,5,6],"inicio":"08:00","fin":"18:00","duracion_min":60}'::jsonb,
    '{
        "tono_marca": "Cercano, profesional y claro. Guia al cliente sin presionar.",
        "direccion": "Estudio privado (confirmar ubicacion exacta al agendar).",
        "metodos_pago": "Transferencia bancaria, Nequi y efectivo (50% al reservar y 50% al entregar).",
        "politicas": "Reprogramacion gratuita con al menos 24 horas de anticipacion. Entrega digital en alta resolucion en 5 dias habiles.",
        "preguntas_frecuentes": [
            {"pregunta": "Entregan las fotos editadas?", "respuesta": "Si, todas las fotos incluidas en cada paquete se entregan editadas profesionalmente en galeria digital."}
        ]
    }'::jsonb,
    'pbkdf2_sha256$sma2026demo00001$530c9063fb99eacf687a4165bdb145b43babafa5eca20f868c6f8944e0aef9e1'
) ON CONFLICT (id) DO NOTHING;

INSERT INTO public.servicios (
    negocio_id, clave, servicio_id_num, nombre, descripcion, tarifa_unitaria,
    palabras_clave, requiere_complejidad, costo_adicional_mantenimiento,
    duracion_minutos, activo
) VALUES
    ('estudio_fotografia', 'sesion_producto', 1, 'Fotografia de Producto E-Commerce', 'Incluye 10 fotos editadas en alta resolucion en estudio, iluminacion profesional y fondo limpio.', 280000, '["producto","catalogo","articulo","ecommerce"]'::jsonb, FALSE, 0, 60, TRUE),
    ('estudio_fotografia', 'retrato_corporativo', 2, 'Retrato Corporativo y Marca Personal', 'Incluye 8 fotos editadas en alta resolucion en estudio (hasta 2 personas) con retoque profesional.', 320000, '["retrato","corporativo","perfil","marca personal"]'::jsonb, FALSE, 0, 60, TRUE),
    ('estudio_fotografia', 'cobertura_evento', 3, 'Cobertura Fotografica de Eventos', 'Incluye 30 fotos editadas en galeria digital (2 a 3 horas en locacion).', 450000, '["evento","boda","grado","fiesta"]'::jsonb, FALSE, 0, 180, TRUE),
    ('estudio_fotografia', 'video_reel', 4, 'Produccion de Video Reels / Comercial Corto', 'Incluye 2 reels 4K editados, colorizacion y musica para redes sociales.', 390000, '["video","reel","reels","comercial"]'::jsonb, FALSE, 0, 120, TRUE)
ON CONFLICT (negocio_id, clave) DO NOTHING;

INSERT INTO public.promociones_vigentes (
    negocio_id, codigo, descripcion, descuento_max, activa, condiciones
) VALUES (
    'estudio_fotografia',
    'PACK_SESIONES_10',
    '10% de descuento al reservar 2 o mas sesiones',
    10.0,
    TRUE,
    '{"cantidad_minima":2,"cantidad_maxima":20}'::jsonb
) ON CONFLICT (negocio_id, codigo) DO NOTHING;

-- =============================================================================
-- SEMILLA 2: mile_cuentas_beauty (Mile Cuentas Beauty - Manicura y Pedicura)
-- PIN inicial de demo = 1234 (cambiar desde el panel PWA)
-- =============================================================================
INSERT INTO public.negocios (
    id, nombre_negocio, rubro, moneda, unidad_cantidad, etiqueta_cantidad,
    etiqueta_complejidad, telefono_dueno, whatsapp_negocio, margen_base, margen_minimo,
    descuento_maximo, mensaje_catalogo_intro, activo, zona_horaria,
    horario_laboral, reglas_y_faq, pin_dueno_hash
) VALUES (
    'mile_cuentas_beauty',
    'Mile Cuentas Beauty',
    'Servicios de Manicura y Pedicura',
    'COP',
    'servicios',
    'cantidad_servicios',
    'diseno',
    '573006900581',
    '573006900581',
    0.45,
    0.45,
    0.0,
    'Con gusto te comparto nuestros servicios de manicura y pedicura',
    TRUE,
    'America/Bogota',
    '{"dias":[1,2,3,4,5,6,7],"inicio":"08:00","fin":"17:00","duracion_min":60,"pausas":[["12:00","14:00"]],"buffer_minutos":15,"max_dias_anticipacion":1}'::jsonb,
    '{
        "propietaria": "Mileidis Cuentas",
        "ciudad": "Cartagena, Colombia",
        "instagram": "@milecuentas_beauty",
        "tono_marca": "Muy calido, cercano, femenino, elegante, natural y servicial. Puedes usar expresiones como reina, mi amor y bella con naturalidad y sin repetirlas en exceso.",
        "direccion": "No se comparte automaticamente. La propietaria envia la ubicacion exacta el dia de la cita.",
        "metodos_pago": "Nequi (llave Bre-B 300 690 0581, titular Mileidis Cuentas), efectivo y tarjetas de credito (recargo del 5% por pago con tarjeta). Enviar comprobante al WhatsApp 300 690 0581.",
        "politicas": "Atencion en local y a domicilio (domicilio desde $15.000 COP segun distancia). Una clienta por cita. Solo se permite combinar 1 servicio de manos y 1 de pies por reserva. Reserva con maximo 1 dia de anticipacion. Anticipo obligatorio del 20% (en diciembre, 1 hora para enviarlo). La cita queda PENDIENTE hasta que la propietaria verifique manualmente el pago. Cancelaciones o inasistencia pierden el anticipo. Reprogramaciones minimo 24 horas antes. Tolerancia maxima de llegada tarde: 15 minutos.",
        "reglas_cotizacion": {
            "una_por_categoria": ["manos", "pies"],
            "costo_diseno_complejo": 25000.0,
            "costo_domicilio_minimo": 15000.0,
            "anticipo_porcentaje": 20.0,
            "recargo_tarjeta_pct": 5.0,
            "requiere_confirmacion_manual_pago": true
        },
        "instrucciones_adicionales": "1. NO envies el catalogo automaticamente al iniciar la conversacion; compartelo solo cuando la clienta lo necesite y SIN listar precios en el catalogo general (comunica precios de forma conversacional al cotizar o cuando pregunten).\n2. Pregunta si desea atencion en el local o a domicilio (si es domicilio, pide sector/ubicacion e informa que el domicilio parte desde $15.000 COP segun distancia). Pregunta tambien si desea diseno simple (incluido) o diseno complejo (+$25.000 COP, pide foto de referencia) y si requiere retiro de trabajo anterior (20 a 60 min extra y hasta $20.000 COP, requiere valoracion humana con foto).\n3. Nunca confirmes una cita como definitiva: al agendar indica el valor del anticipo del 20%, los datos de pago (Bre-B / Nequi 300 690 0581 a nombre de Mileidis Cuentas) y aclara que la cita queda pendiente hasta que la propietaria verifique el pago.\n4. Si preguntan por servicios de cabello (hidratacion capilar, blower, keratinas, tintes) o cualquier servicio fuera de manicura/pedicura, NO cotices ni agendes: responde con calidez que la propietaria le dara esa informacion y llama a escalar_a_humano.\n5. Escala tambien a escalar_a_humano si no es claro si el diseno es simple o complejo, si requiere valorar retiro, si pide excepcion a politicas o si hay queja.",
        "preguntas_frecuentes": [
            {"pregunta": "Donde estan ubicados?", "respuesta": "Estamos en Cartagena. Por privacidad, Mileidis te envia la ubicacion exacta el dia de tu cita."},
            {"pregunta": "Tienen servicio a domicilio?", "respuesta": "Si, ofrecemos domicilio para todos los servicios desde $15.000 COP segun la distancia."},
            {"pregunta": "El diseno tiene costo adicional?", "respuesta": "Los diseños simples del portafolio estan incluidos en el precio base. Los diseños complejos (dibujos elaborados en todas las uñas) tienen un adicional de $25.000 COP."},
            {"pregunta": "Que pasa si tengo trabajo anterior para retirar?", "respuesta": "El retiro toma entre 20 y 60 minutos adicionales y puede costar hasta $20.000 COP segun el trabajo; debemos valorarlo con una foto de tus uñas actuales."}
        ]
    }'::jsonb,
    'pbkdf2_sha256$sma2026demo00001$530c9063fb99eacf687a4165bdb145b43babafa5eca20f868c6f8944e0aef9e1'
) ON CONFLICT (id) DO NOTHING;

INSERT INTO public.servicios (
    negocio_id, clave, servicio_id_num, nombre, descripcion, tarifa_unitaria,
    palabras_clave, requiere_complejidad, costo_adicional_mantenimiento,
    duracion_minutos, activo
) VALUES
    ('mile_cuentas_beauty', 'manicura_semipermanente', 1, 'Manicura Semipermanente', 'Color y brillo duradero. Limpieza, preparacion y esmaltado semipermanente.', 35000, '["manos","manicura","semipermanente","unas"]'::jsonb, FALSE, 0, 60, TRUE),
    ('mile_cuentas_beauty', 'kapping_gel', 2, 'Kapping Gel / Acrilico / Rubber', 'Fortalecimiento de una natural. Proteccion y resistencia sin perder naturalidad.', 45000, '["manos","manicura","kapping","acrilico","rubber"]'::jsonb, FALSE, 0, 60, TRUE),
    ('mile_cuentas_beauty', 'dual_gel_premium', 3, 'Dual Gel Premium', 'Esculpido Dual Gel. Unas resistentes con acabado impecable y ligero.', 60000, '["manos","manicura","dual gel","esculpido"]'::jsonb, FALSE, 0, 90, TRUE),
    ('mile_cuentas_beauty', 'soft_gel_premium', 4, 'Soft Gel Premium', 'Extensiones Soft Gel. Acabado ligero, elegante y de aspecto natural.', 70000, '["manos","manicura","soft gel","extensiones"]'::jsonb, FALSE, 0, 90, TRUE),
    ('mile_cuentas_beauty', 'pedi_spa_essential', 5, 'Pedi Spa Essential', 'Pedicure Spa tradicional. Limpieza de cuticulas, preparacion de unas y esmaltado semipermanente.', 25000, '["pies","pedicura","pedi spa","essential"]'::jsonb, FALSE, 0, 60, TRUE),
    ('mile_cuentas_beauty', 'pedi_spa_care', 6, 'Pedi Spa Care', 'Pedicure Spa medium. Limpieza profunda, retiro de callosidades, hidratacion y esmaltado semipermanente.', 37000, '["pies","pedicura","pedi spa","care","callosidades"]'::jsonb, FALSE, 0, 60, TRUE),
    ('mile_cuentas_beauty', 'pedi_spa_luxury', 7, 'Pedi Spa Luxury', 'Pedicure Spa premium. Limpieza profunda, retiro de callosidades, hidratacion intensiva, terapia con burbuja efervescente, masaje con vela caliente y esmaltado semipermanente.', 50000, '["pies","pedicura","pedi spa","luxury","vela caliente"]'::jsonb, FALSE, 0, 90, TRUE)
ON CONFLICT (negocio_id, clave) DO NOTHING;
