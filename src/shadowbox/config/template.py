"""カードテンプレート定義モジュール。

このモジュールは、TCGカードのイラスト領域を定義するための
データクラスを提供します。テンプレートはYAMLファイルとして
保存・読み込みが可能です。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """イラスト領域の座標を表すバウンディングボックス。

    ピクセル単位で矩形領域を定義します。
    frozen=Trueにより不変オブジェクトとして扱われます。

    Attributes:
        x: 左端のX座標 (ピクセル)。
        y: 上端のY座標 (ピクセル)。
        width: 幅 (ピクセル)。
        height: 高さ (ピクセル)。

    Example:
        >>> # ポケモンカードのイラスト領域
        >>> bbox = BoundingBox(x=42, y=72, width=660, height=488)
        >>> print(f"右端: {bbox.right}, 下端: {bbox.bottom}")
        右端: 702, 下端: 560
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """初期化後のバリデーション。

        Raises:
            ValueError: 幅・高さが正でない場合、またはx・yが負の場合。
        """
        if self.width <= 0:
            raise ValueError(f"幅は正の値である必要があります。取得値: {self.width}")
        if self.height <= 0:
            raise ValueError(f"高さは正の値である必要があります。取得値: {self.height}")
        if self.x < 0:
            raise ValueError(f"xは0以上である必要があります。取得値: {self.x}")
        if self.y < 0:
            raise ValueError(f"yは0以上である必要があります。取得値: {self.y}")

    @property
    def right(self) -> int:
        """右端のX座標を返す。"""
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """下端のY座標を返す。"""
        return self.y + self.height

    def to_tuple(self) -> tuple[int, int, int, int]:
        """(x, y, width, height)のタプルとして返す。

        Returns:
            座標と寸法のタプル。
        """
        return (self.x, self.y, self.width, self.height)

    def to_crop_box(self) -> tuple[int, int, int, int]:
        """PILのcrop用ボックス形式で返す。

        Returns:
            (left, upper, right, lower)のタプル。
        """
        return (self.x, self.y, self.right, self.bottom)


@dataclass(frozen=True)
class CardTemplate:
    """特定のカードゲーム用テンプレート。

    カードの寸法とイラスト領域の位置を定義します。
    テンプレートを作成しておくことで、同じゲームのカードに
    対して繰り返し同じ領域を適用できます。

    Attributes:
        name: テンプレート名 (例: "pokemon_standard")。
        game: ゲーム名 (例: "pokemon", "mtg", "yugioh")。
        illustration_area: イラスト領域のバウンディングボックス。
        card_width: カードの幅 (ピクセル)。
        card_height: カードの高さ (ピクセル)。
        frame_margin: フレームエフェクト用のマージン (ピクセル)。
        description: テンプレートの説明 (オプション)。

    Example:
        >>> # ポケモンカード用テンプレートの作成
        >>> template = CardTemplate(
        ...     name="pokemon_standard",
        ...     game="pokemon",
        ...     illustration_area=BoundingBox(x=42, y=72, width=660, height=488),
        ...     card_width=744,
        ...     card_height=1039,
        ...     description="標準的なポケモンカードのイラスト領域"
        ... )
    """

    name: str
    game: str
    illustration_area: BoundingBox
    card_width: int
    card_height: int
    frame_margin: int = 10
    description: str | None = None

    def __post_init__(self) -> None:
        """初期化後のバリデーション。

        イラスト領域がカードの範囲内に収まっているか検証します。

        Raises:
            ValueError: カード寸法が不正、またはイラスト領域がはみ出している場合。
        """
        if self.card_width <= 0:
            raise ValueError(
                f"card_widthは正の値である必要があります。取得値: {self.card_width}"
            )
        if self.card_height <= 0:
            raise ValueError(
                f"card_heightは正の値である必要があります。取得値: {self.card_height}"
            )
        if self.frame_margin < 0:
            raise ValueError(
                f"frame_marginは0以上である必要があります。取得値: {self.frame_margin}"
            )

        # イラスト領域がカード内に収まっているか検証
        bbox = self.illustration_area
        if bbox.right > self.card_width:
            raise ValueError(
                f"イラスト領域がカードの幅を超えています: "
                f"{bbox.right} > {self.card_width}"
            )
        if bbox.bottom > self.card_height:
            raise ValueError(
                f"イラスト領域がカードの高さを超えています: "
                f"{bbox.bottom} > {self.card_height}"
            )

    def to_dict(self) -> dict:
        """YAMLシリアライズ用の辞書に変換。

        Returns:
            テンプレートデータを含む辞書。
        """
        return {
            "name": self.name,
            "game": self.game,
            "card_width": self.card_width,
            "card_height": self.card_height,
            "frame_margin": self.frame_margin,
            "description": self.description,
            "illustration_area": {
                "x": self.illustration_area.x,
                "y": self.illustration_area.y,
                "width": self.illustration_area.width,
                "height": self.illustration_area.height,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CardTemplate":
        """辞書からテンプレートを作成 (YAMLデシリアライズ用)。

        Args:
            data: テンプレートデータを含む辞書。

        Returns:
            CardTemplateインスタンス。
        """
        bbox_data = data["illustration_area"]
        bbox = BoundingBox(
            x=bbox_data["x"],
            y=bbox_data["y"],
            width=bbox_data["width"],
            height=bbox_data["height"],
        )
        return cls(
            name=data["name"],
            game=data["game"],
            illustration_area=bbox,
            card_width=data["card_width"],
            card_height=data["card_height"],
            frame_margin=data.get("frame_margin", 10),
            description=data.get("description"),
        )
