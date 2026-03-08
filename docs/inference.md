# Inference / Инференс

## `predict_track`

EN: Use when you already have a track tensor.

RU: Используйте, когда у вас уже есть track tensor.

## `predict_image`

EN: Use when you have one image and want to convert it to a pseudo-track.

RU: Используйте, когда у вас есть одна картинка и её нужно преобразовать в псевдо-трек.

## `predict_images`

EN: Use when you have a batch of still images and want one pseudo-track per image.

RU: Используйте, когда у вас есть батч статических изображений и нужен один псевдо-трек на каждую картинку.

## `predict_group`

EN: Use when you have several related images describing one observation, for example several views of one object or several synchronized camera snapshots.

RU: Используйте, когда у вас есть несколько связанных изображений, описывающих одно наблюдение, например несколько ракурсов одного объекта или несколько синхронных снимков с разных камер.

EN: This helper does not apply the notebook gold protocol. If the group is shorter than the active third, the missing positions can be filled either by repeating images or by applying `active_frame_transform` to the repeated part.

RU: Этот helper не применяет notebook gold-протокол. Если группа короче активной трети, недостающие позиции могут быть заполнены либо повторениями изображений, либо с помощью `active_frame_transform`, применённого к дополняемой части.

## `predict_video`

EN: Use when you have a video file or a decoded frame sequence and want to sample it into a track.

RU: Используйте, когда у вас есть видеофайл или уже декодированная последовательность кадров, которую нужно превратить в трек.

EN: By default the video adapter uses uniform sampling across the clip. If the clip is shorter than the active third, the missing positions can be synthesized through repetition or repetition with `active_frame_transform`.

RU: По умолчанию видео-адаптер использует равномерный sampling по всему ролику. Если ролик короче активной трети, недостающие позиции можно достроить повторением или повторением с `active_frame_transform`.

## `apply_gold`

EN: `apply_gold=True` applies the notebook-compatible gold protocol before forward. This is appropriate for a single image converted to a pseudo-track, but not for grouped-image or video inference.

RU: `apply_gold=True` применяет notebook-compatible gold-протокол перед forward. Это уместно для одной картинки, преобразованной в pseudo-track, но не для группового или видео-инференса.
