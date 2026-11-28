-- =============================================================================
-- Plataforma Comercial Multi-Negocio (Rama: comercial)
-- Migracion aditiva segura para Supabase PostgreSQL (compatible con esquema base)
-- Incluye RLS estricto, indices UNIQUE por canal y perfiles semilla iniciales
-- =============================================================================

-- 1. Columnas multi-negocio, horarios, FAQ, WhatsApp Cloud API y acceso por dueno
ALTER TABLE negocios
    ADD COLUMN IF NOT EXISTS activo BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS whatsapp_negocio TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS whatsapp_phone_number_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS whatsapp_access_token TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS botpress_bot_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS google_calendar_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS zona_horaria TEXT NOT NULL DEFAULT 'America/Bogota',
    ADD COLUMN IF NOT EXISTS horario_laboral JSONB NOT NULL DEFAULT '{"dias":[1,2,3,4,5,6],"inicio":"08:00","fin":"18:00","duracion_min":60}'::jsonb,
    ADD COLUMN IF NOT EXISTS reglas_y_faq JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS pin_dueno_hash TEXT NOT NULL DEFAULT '';

-- 2. Duracion y estado activo en catalogo de servicios
ALTER TABLE servicios
    ADD COLUMN IF NOT EXISTS duracion_minutos INTEGER NOT NULL DEFAULT 60,
    ADD COLUMN IF NOT EXISTS activo BOOLEAN NOT NULL DEFAULT TRUE;

-- 3. Timestamp de actualizacion en perfiles de clientes y resumen IA en alertas
ALTER TABLE perfiles_clientes
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

ALTER TABLE alertas_humanas
    ADD COLUMN IF NOT EXISTS resumen_ia TEXT NOT NULL DEFAULT '';

-- 4. Indices UNIQUE parciales para evitar colision / secuestro de canales WhatsApp/Botpress
DROP INDEX IF EXISTS idx_negocios_wa_phone_id;
CREATE UNIQUE INDEX IF NOT EXISTS idx_negocios_wa_phone_id
    ON negocios (whatsapp_phone_number_id)
    WHERE whatsapp_phone_number_id <> '';

DROP INDEX IF EXISTS idx_negocios_wa_numero;
CREATE UNIQUE INDEX IF NOT EXISTS idx_negocios_wa_numero
    ON negocios (whatsapp_negocio)
    WHERE whatsapp_negocio <> '';

CREATE UNIQUE INDEX IF NOT EXISTS idx_negocios_botpress_id
    ON negocios (botpress_bot_id)
    WHERE botpress_bot_id <> '';

CREATE INDEX IF NOT EXISTS idx_alertas_negocio_abiertas
    ON alertas_humanas (negocio_id, resuelta, id DESC);

-- 5. Seguridad multi-tenant: Row Level Security (RLS) obligatorio en todas las tablas
-- Sin politicas publicas para anon/authenticated; el backend accede con service_role / secret key.
ALTER TABLE public.negocios ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.servicios ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.promociones_vigentes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.perfiles_clientes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alertas_humanas ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agendamientos ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.tarifas_historicas ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.historico_ofertas ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- SEMILLA 1: Actualizacion comercial de estudio_fotografia
-- PIN inicial de demo = 1234 (OBLIGATORIO cambiar desde el panel PWA al desplegar)
-- =============================================================================
UPDATE negocios
SET
    activo = TRUE,
    zona_horaria = 'America/Bogota',
    horario_laboral = '{"dias":[1,2,3,4,5,6],"inicio":"08:00","fin":"18:00","duracion_min":60}'::jsonb,
    reglas_y_faq = '{
        "tono_marca": "Cercano, profesional y claro. Guia al cliente sin presionar.",
        "direccion": "Estudio privado (confirmar ubicacion exacta al agendar).",
        "metodos_pago": "Transferencia bancaria, Nequi y efectivo (50% al reservar y 50% al entregar).",
        "politicas": "Reprogramacion gratuita con al menos 24 horas de anticipacion. Entrega digital en alta resolucion en 5 dias habiles.",
        "preguntas_frecuentes": [
            {"pregunta": "Entregan las fotos editadas?", "respuesta": "Si, todas las fotos incluidas en cada paquete se entregan editadas profesionalmente en galeria digital."}
        ]
    }'::jsonb,
    pin_dueno_hash = CASE
        WHEN pin_dueno_hash = '' THEN 'pbkdf2_sha256$sma2026demo00001$530c9063fb99eacf687a4165bdb145b43babafa5eca20f868c6f8944e0aef9e1'
        ELSE pin_dueno_hash
    END
WHERE id = 'estudio_fotografia';

UPDATE servicios
SET duracion_minutos = CASE clave
    WHEN 'cobertura_evento' THEN 180
    WHEN 'video_reel' THEN 120
    ELSE 60
END,
activo = TRUE
WHERE negocio_id = 'estudio_fotografia';

-- =============================================================================
-- SEMILLA 2: Negocio independiente de belleza (salon_belleza)
-- PIN inicial de demo = 1234 (OBLIGATORIO cambiar desde el panel PWA al desplegar)
-- =============================================================================
INSERT INTO negocios (
    id, nombre_negocio, rubro, moneda, unidad_cantidad, etiqueta_cantidad,
    etiqueta_complejidad, telefono_dueno, margen_base, margen_minimo,
    descuento_maximo, mensaje_catalogo_intro, activo, zona_horaria,
    horario_laboral, reglas_y_faq, pin_dueno_hash
) VALUES (
    'salon_belleza',
    'Atelier Belleza y Estilo',
    'Salon de Belleza, Cabello, Unas y Cejas',
    'COP',
    'servicios',
    'cantidad_servicios',
    'complejidad',
    '',
    0.45,
    0.25,
    10.0,
    'Hola! Con gusto te comparto nuestros servicios de belleza y cuidado personal',
    TRUE,
    'America/Bogota',
    '{"dias":[2,3,4,5,6],"inicio":"09:00","fin":"19:00","duracion_min":60}'::jsonb,
    '{
        "tono_marca": "Calido, amable y atento, ideal para atencion en salon de belleza.",
        "direccion": "Sede principal (actualizar direccion exacta desde el panel del negocio).",
        "metodos_pago": "Nequi, Daviplata, transferencia bancaria y efectivo en el salon.",
        "politicas": "Llegar 10 minutos antes de la cita. Cambios o cancelaciones con minimo 4 horas de anticipacion.",
        "preguntas_frecuentes": [
            {"pregunta": "Atienden los lunes?", "respuesta": "Nuestro horario regular es de martes a sabado de 9:00 a. m. a 7:00 p. m."}
        ]
    }'::jsonb,
    'pbkdf2_sha256$sma2026demo00001$530c9063fb99eacf687a4165bdb145b43babafa5eca20f868c6f8944e0aef9e1'
) ON CONFLICT (id) DO NOTHING;

INSERT INTO servicios (
    negocio_id, clave, servicio_id_num, nombre, descripcion, tarifa_unitaria,
    palabras_clave, requiere_complejidad, costo_adicional_mantenimiento,
    duracion_minutos, activo
) VALUES
    ('salon_belleza', 'manicure_semipermanente', 1, 'Manicure Semipermanente', 'Limpieza completa, hidratacion de cuticula y esmaltado semipermanente de larga duracion.', 45000, '["manicure","unas","semipermanente","manos"]'::jsonb, FALSE, 0, 60, TRUE),
    ('salon_belleza', 'pedicure_spa', 2, 'Pedicure Spa Completo', 'Exfoliacion, limpieza profunda, hidratacion spa y esmaltado tradicional o semipermanente.', 55000, '["pedicure","pies","spa"]'::jsonb, FALSE, 0, 60, TRUE),
    ('salon_belleza', 'diseno_cejas_laminado', 3, 'Diseno de Cejas y Laminado', 'Diseno segun morfologia del rostro, depilacion con hilo o cera y laminado nutritivo.', 65000, '["cejas","laminado","depilacion","pestanas"]'::jsonb, FALSE, 0, 45, TRUE),
    ('salon_belleza', 'cepillado_hidratacion', 4, 'Cepillado y Tratamiento de Hidratacion', 'Lavado detox, mascarilla nutritiva de keratina o colageno y cepillado profesional.', 75000, '["cepillado","hidratacion","cabello","pelo","blower"]'::jsonb, FALSE, 0, 90, TRUE),
    ('salon_belleza', 'balayage_iluminacion', 5, 'Balayage e Iluminacion Personalizada', 'Diseno de color a medida con proteccion capilar, matizacion y finalizacion con ondas.', 260000, '["balayage","color","tintura","iluminaciones","mechas"]'::jsonb, FALSE, 0, 180, TRUE)
ON CONFLICT (negocio_id, clave) DO NOTHING;

INSERT INTO promociones_vigentes (
    negocio_id, codigo, descripcion, descuento_max, activa, condiciones
) VALUES (
    'salon_belleza',
    'COMBO_BELLEZA_10',
    '10% de descuento al reservar 2 o mas servicios en la misma visita',
    10.0,
    TRUE,
    '{"servicios":["manicure_semipermanente","pedicure_spa","diseno_cejas_laminado","cepillado_hidratacion"],"cantidad_minima":2,"cantidad_maxima":10}'::jsonb
) ON CONFLICT (negocio_id, codigo) DO NOTHING;
