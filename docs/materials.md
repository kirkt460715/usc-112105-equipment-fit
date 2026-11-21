# 辅具借用换配资料约定

合成借用档案，尺寸为厘米，仅用于借用点的库存和人工测量管理。

## loans

借用当日人员、站点、用途和尺寸资料。needs 列不可缺少的配件类别。previous 为一项续借或换配所由来的上一笔，borrow_end 与 due 分别是正常归还和复核期限。

字段 `loan`, `person`, `site`, `borrowed_on`, `borrow_end`, `due`, `use`, `reach`, `doorway`, `needs`, `previous`, `protected`。

## outcomes

借出以后的适配复核和归还理由；无结果或晚到不能自动作成功标签。

字段 `reason`, `loan`, `observed_on`, `kind`, `evidence`。

## reviews

对既有复核理由的替换日期以及复核时才得知的尺寸，不能写回借出时的特征。

字段 `revision`, `reason`, `observed_on`, `replacement`, `corrected_reach`。

## devices

可调主件及合规用途、承重之外的厘米调整范围和宽度。size 为器具能覆盖的用户尺寸，width 为通过空间所需最小净宽。

字段 `device`, `family`, `site`, `uses`, `size_low`, `size_high`, `width`。

## parts

独立配件编号与可用于哪些器具系列。checked 表示已确认可再次入库。

字段 `part`, `category`, `families`, `checked`。

## requests

下轮待换配个案、原主件及仍借用配件、是否由人员确认、预计剩余使用日、可到场日期和实际能交回主件的日期。旧承诺优先，之后按保护标识、适配确证、使用日和可信风险依序比较可实际兑现的安排。

字段 `request`, `loan`, `old_device`, `old_parts`, `verified`, `remaining_days`, `exchange_dates`, `can_return_on`。

## reservations

已签收的新配置、旧件及有效期限。主件与配件都预留。

字段 `reservation`, `request`, `device`, `parts`, `until_on`, `state`。

## stock_events

记录用户归还、清点、到场三类事实。归还日期与可重用日期可能不同。

字段 `event`, `observed_on`, `asset`, `kind`, `request`。
