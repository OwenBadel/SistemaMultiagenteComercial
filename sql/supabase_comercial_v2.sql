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

-- =============================================================================
-- SEMILLA 3: Mile Cuentas Beauty (Cartagena, Manicura y Pedicura - Fase 1)
-- PIN inicial de demo = 1234 (OBLIGATORIO cambiar desde el panel PWA al desplegar)
-- =============================================================================
INSERT INTO negocios (
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

INSERT INTO servicios (
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
